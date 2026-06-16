from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .retrieve import RetrievalConfig, Retriever
from .storage import read_jsonl, write_json


def evaluate_retrieval(
    index_path: Path,
    qa_path: Path,
    top_k: int = 10,
    mode: str = "hybrid",
    calibrated: bool = True,
    limit: int | None = None,
    retrieval_config: RetrievalConfig | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval against QA evidence documents and return aggregate metrics."""
    retriever = Retriever(index_path, calibrated=calibrated, config=retrieval_config)
    qa_rows = read_jsonl(qa_path)[:limit]
    recalls: list[float] = []
    precisions: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    all_evidence_successes: list[float] = []
    packed_context_recalls: list[float] = []
    latencies: list[float] = []
    answer_hits: list[float] = []
    skipped_without_evidence = 0
    skipped_null_queries = 0
    skipped_incomplete_evidence = 0
    null_queries = 0
    null_abstention_hits = 0
    by_type: defaultdict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"recall_at_k": [], "precision_at_k": [], "mrr": []}
    )
    details: list[dict[str, Any]] = []
    for row in qa_rows:
        query = str(row.get("query") or row.get("question") or "")
        answer = str(row.get("answer") or "")
        if not query:
            continue
        result = retriever.retrieve(query, mode=mode, top_k=top_k, token_budget=1600)
        contexts = result["context"]
        gold_document_ids, expected_evidence_count = _gold_document_ids(row, retriever.documents)
        if not gold_document_ids or len(gold_document_ids) < expected_evidence_count:
            skipped_without_evidence += 1
            is_null = expected_evidence_count == 0
            skipped_null_queries += is_null
            skipped_incomplete_evidence += not is_null
            if is_null:
                null_queries += 1
                null_abstention_hits += int(bool(result.get("abstained", False)))
            details.append({
                "id": row.get("id"),
                "query": query,
                "evaluated": False,
                "reason": "null query without evidence" if is_null else "incomplete evidence documents in index",
                "expected_evidence_count": expected_evidence_count,
                "indexed_evidence_count": len(gold_document_ids),
                "top_context_ids": [item["id"] for item in contexts],
            })
            continue
        retrieved_texts = [str(item["text"]) for item in contexts]
        packed_context_document_ids = [
            set(item.get("metadata", {}).get("document_ids", []))
            for item in contexts
        ]
        packed_document_ids = (
            set().union(*packed_context_document_ids) if packed_context_document_ids else set()
        )
        ranked_documents = result.get("ranked_documents", [])
        if ranked_documents:
            ranked_document_ids = [str(item["id"]) for item in ranked_documents[:top_k]]
            ranked_document_sets = [{document_id} for document_id in ranked_document_ids]
            retrieved_document_ids = set(ranked_document_ids)
        else:
            ranked_document_sets = packed_context_document_ids
            retrieved_document_ids = packed_document_ids
        hits = [bool(document_ids & gold_document_ids) for document_ids in ranked_document_sets]
        recall = len(retrieved_document_ids & gold_document_ids) / len(gold_document_ids)
        precision = len(retrieved_document_ids & gold_document_ids) / max(1, len(retrieved_document_ids))
        all_evidence_success = float(gold_document_ids <= retrieved_document_ids)
        packed_context_recall = len(packed_document_ids & gold_document_ids) / len(gold_document_ids)
        mrr = 0.0
        for idx, hit in enumerate(hits, start=1):
            if hit:
                mrr = 1.0 / idx
                break
        dcg = sum((1.0 / math.log2(index + 2)) for index, hit in enumerate(hits) if hit)
        ideal_hits = min(len(gold_document_ids), top_k)
        ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
        answer_evaluable = answer.casefold() not in {"yes", "no", "insufficient information."}
        answer_hit = 1.0 if answer_evaluable and any(_contains_answer(text, answer) for text in retrieved_texts) else 0.0
        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)
        ndcgs.append(ndcg)
        all_evidence_successes.append(all_evidence_success)
        packed_context_recalls.append(packed_context_recall)
        latencies.append(float(result.get("diagnostics", {}).get("total_ms", 0.0)))
        question_type = str(row.get("metadata", {}).get("question_type") or "unknown")
        by_type[question_type]["recall_at_k"].append(recall)
        by_type[question_type]["precision_at_k"].append(precision)
        by_type[question_type]["mrr"].append(mrr)
        by_type[question_type].setdefault("all_evidence_success_at_k", []).append(all_evidence_success)
        by_type[question_type].setdefault("ndcg_at_k", []).append(ndcg)
        if answer_evaluable:
            answer_hits.append(answer_hit)
        details.append({
            "id": row.get("id"),
            "query": query,
            "evaluated": True,
            "gold_document_ids": sorted(gold_document_ids),
            "retrieved_document_ids": sorted(retrieved_document_ids),
            "recall_at_k": recall,
            "precision_at_k": precision,
            "mrr": mrr,
            "ndcg_at_k": ndcg,
            "all_evidence_success_at_k": all_evidence_success,
            "packed_context_recall_at_k": packed_context_recall,
            "answer_hit": answer_hit,
            "answer_evaluable": answer_evaluable,
            "top_context_ids": [item["id"] for item in contexts],
            "ranked_document_ids": [item["id"] for item in ranked_documents],
            "retrieval_diagnostics": result.get("diagnostics", {}),
        })
    return {
        "index": str(index_path),
        "qa": str(qa_path),
        "mode": mode,
        "calibrated": calibrated,
        "top_k": top_k,
        "count": len(recalls),
        "total_qa": len(qa_rows),
        "skipped_without_evidence": skipped_without_evidence,
        "skipped_null_queries": skipped_null_queries,
        "skipped_incomplete_evidence": skipped_incomplete_evidence,
        "null_abstention_accuracy": round(null_abstention_hits / null_queries, 6) if null_queries else 0.0,
        "recall_at_k": round(mean(recalls), 6) if recalls else 0.0,
        "precision_at_k": round(mean(precisions), 6) if precisions else 0.0,
        "mrr": round(mean(mrrs), 6) if mrrs else 0.0,
        "ndcg_at_k": round(mean(ndcgs), 6) if ndcgs else 0.0,
        "all_evidence_success_at_k": round(mean(all_evidence_successes), 6) if all_evidence_successes else 0.0,
        "packed_context_recall_at_k": round(mean(packed_context_recalls), 6) if packed_context_recalls else 0.0,
        "answer_hit_rate": round(mean(answer_hits), 6) if answer_hits else 0.0,
        "answer_evaluated_count": len(answer_hits),
        "latency_ms_p50": round(_percentile(latencies, 0.50), 3),
        "latency_ms_p95": round(_percentile(latencies, 0.95), 3),
        "by_question_type": {
            question_type: {
                metric: round(mean(values), 6) if values else 0.0
                for metric, values in metrics.items()
            } | {"count": len(metrics["mrr"])}
            for question_type, metrics in by_type.items()
        },
        "details": details,
    }


def compare_baseline_calibrated(
    index_path: Path,
    qa_path: Path,
    top_k: int = 10,
    mode: str = "hybrid",
    limit: int | None = None,
    retrieval_config: RetrievalConfig | None = None,
) -> dict[str, Any]:
    """Compare retrieval with and without calibrated edge reliability."""
    baseline = evaluate_retrieval(
        index_path, qa_path, top_k, mode, calibrated=False, limit=limit,
        retrieval_config=retrieval_config,
    )
    calibrated = evaluate_retrieval(
        index_path, qa_path, top_k, mode, calibrated=True, limit=limit,
        retrieval_config=retrieval_config,
    )
    summary = {
        "baseline": {k: v for k, v in baseline.items() if k != "details"},
        "calibrated": {k: v for k, v in calibrated.items() if k != "details"},
        "delta": {
            key: round(calibrated[key] - baseline[key], 6)
            for key in (
                "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k",
                "all_evidence_success_at_k", "answer_hit_rate",
            )
        },
    }
    return {"summary": summary, "baseline_details": baseline["details"], "calibrated_details": calibrated["details"]}


def failed_evidence_units(index_path: Path, qa_path: Path) -> list[dict[str, Any]]:
    """Return unresolved LLM extraction errors that affect gold evidence documents."""
    errors = read_jsonl(index_path / "llm_errors.jsonl")
    if not errors:
        return []
    documents = {document.id: document for document in Retriever(index_path).documents}
    qa_rows = read_jsonl(qa_path)
    evidence_urls = {
        str(item.get("url"))
        for row in qa_rows
        for item in row.get("evidence", [])
        if isinstance(item, dict) and item.get("url")
    }
    evidence_titles = {
        str(item.get("title")).casefold()
        for row in qa_rows
        for item in row.get("evidence", [])
        if isinstance(item, dict) and item.get("title")
    }
    result = []
    for error in errors:
        document = documents.get(str(error.get("document_id")))
        if not document:
            continue
        metadata = document.metadata
        nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
        url = str(nested.get("url") or metadata.get("id") or "")
        title = str(metadata.get("title") or nested.get("title") or document.title).casefold()
        if url in evidence_urls or title in evidence_titles:
            result.append(error)
    return result


def _gold_document_ids(row: dict[str, Any], documents: list[Any]) -> tuple[set[str], int]:
    """Resolve QA evidence references to indexed document identifiers."""
    evidence = row.get("evidence") or []
    evidence_urls = {
        str(item.get("url"))
        for item in evidence
        if isinstance(item, dict) and item.get("url")
    }
    evidence_titles = {
        str(item.get("title")).casefold()
        for item in evidence
        if isinstance(item, dict) and item.get("title")
    }
    result: set[str] = set()
    for document in documents:
        metadata = document.metadata
        original_id = str(metadata.get("id") or "")
        nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
        url = str(nested.get("url") or original_id)
        title = str(metadata.get("title") or nested.get("title") or document.title).casefold()
        if url in evidence_urls or title in evidence_titles:
            result.add(document.id)
    expected = len(evidence_urls) if evidence_urls else len(evidence_titles)
    return result, expected


def _contains_answer(text: str, answer: str) -> bool:
    if not answer:
        return False
    answer_clean = re.sub(r"\s+", " ", answer.casefold()).strip()
    return bool(answer_clean and answer_clean in text.casefold())


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def main() -> None:
    """Run baseline/calibrated retrieval metrics from the CLI."""
    parser = argparse.ArgumentParser(prog="ecgraphrag.metrics")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", default="two_stage", choices=["heuristic", "embedding", "hybrid", "two_stage"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    result = compare_baseline_calibrated(
        args.index,
        args.qa,
        args.top_k,
        args.mode,
        args.limit,
        RetrievalConfig.from_path(args.config),
    )
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
