from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import evaluate_retrieval, failed_evidence_units
from .retrieve import RetrievalConfig
from .storage import write_json


SUMMARY_KEYS = [
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "all_evidence_success_at_k",
    "packed_context_recall_at_k",
    "answer_hit_rate",
    "null_abstention_accuracy",
    "count",
    "total_qa",
    "latency_ms_p50",
    "latency_ms_p95",
]

_BASE_ABLATION_VARIANTS: list[tuple[str, bool, bool, tuple[str, ...]]] = [
    ("baseline_base_index", False, False, ()),
    ("calibrated_base_index", False, True, ()),
    ("enriched_control_no_fields", True, True, ()),
    ("questions", True, True, ("questions",)),
    ("summaries", True, True, ("summaries",)),
    ("entities", True, True, ("entities",)),
    ("inferred_edges", True, True, ("inferred_edges",)),
    ("importance", True, True, ("importance",)),
    ("contradiction_penalty", True, True, ("contradiction_penalty",)),
    ("questions+summaries", True, True, ("questions", "summaries")),
    ("questions+summaries+entities", True, True, ("questions", "summaries", "entities")),
    ("questions+summaries+importance", True, True, ("questions", "summaries", "importance")),
    (
        "questions+summaries+entities+inferred_edges",
        True,
        True,
        ("questions", "summaries", "entities", "inferred_edges"),
    ),
    (
        "all_enrichment_fields",
        True,
        True,
        ("questions", "summaries", "entities", "inferred_edges", "importance", "contradiction_penalty"),
    ),
]

ABLATION_VARIANTS: list[tuple[str, bool, bool, tuple[str, ...], str]] = [
    (*variant, "two_stage") for variant in _BASE_ABLATION_VARIANTS
] + [
    (f"iterative_{name}", use_enriched, calibrated, signals, "iterative")
    for name, use_enriched, calibrated, signals in _BASE_ABLATION_VARIANTS
]


def run_benchmark(
    index_path: Path,
    qa_path: Path,
    output_path: Path | None = None,
    enriched_index_path: Path | None = None,
    config: RetrievalConfig | None = None,
    top_k: int = 10,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate baseline, calibrated, and optional enriched two-stage retrieval."""
    config = config or RetrievalConfig()
    variants = {
        "baseline_two_stage": (index_path, False),
        "calibrated_two_stage": (index_path, True),
    }
    if enriched_index_path:
        variants["enriched_two_stage"] = (enriched_index_path, True)
    results = {
        name: evaluate_retrieval(
            path,
            qa_path,
            top_k=top_k,
            mode="two_stage",
            calibrated=calibrated,
            limit=limit,
            retrieval_config=config,
        )
        for name, (path, calibrated) in variants.items()
    }
    value = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "qa": str(qa_path),
        "config": asdict(config),
        "summary": {
            name: {key: result[key] for key in SUMMARY_KEYS}
            for name, result in results.items()
        },
        "details": {name: result["details"] for name, result in results.items()},
    }
    if output_path:
        write_json(output_path, value)
    return value


def run_retrieval_ablation(
    dataset_name: str,
    data_path: Path,
    index_path: Path,
    enriched_index_path: Path,
    output_path: Path,
    config: RetrievalConfig | None = None,
    top_k: int = 10,
    limit: int | None = None,
    strict_failed_evidence: bool = True,
) -> dict[str, Any]:
    """Evaluate retrieval variants over one full enriched index."""
    config = config or RetrievalConfig()
    qa_path = data_path / "qa.jsonl"
    if not qa_path.exists():
        raise FileNotFoundError(f"QA file not found: {qa_path}")
    _require_index(index_path)
    _require_index(enriched_index_path)
    output_path.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    calibrated_reference: dict[str, Any] | None = None
    for name, use_enriched_index, calibrated, signals, mode in ABLATION_VARIANTS:
        variant_index = enriched_index_path if use_enriched_index else index_path
        variant_config = _ablation_config(config, signals)
        failures = failed_evidence_units(variant_index, qa_path)
        if failures and strict_failed_evidence:
            summary = _invalid_summary(name, variant_index, calibrated, signals, mode, failures)
            variant_details = []
        else:
            result = evaluate_retrieval(
                variant_index,
                qa_path,
                top_k=top_k,
                mode=mode,
                calibrated=calibrated,
                limit=limit,
                retrieval_config=variant_config,
            )
            summary = {key: result[key] for key in SUMMARY_KEYS}
            summary.update({
                "index": str(variant_index),
                "calibrated": calibrated,
                "retrieval_mode": mode,
                "signals": list(signals),
                "valid": True,
                "failed_evidence_units": len(failures),
                "warning": "failed evidence chunks present" if failures else "",
            })
            variant_details = result["details"]
        summaries[name] = summary
        details[name] = variant_details
        if name == "calibrated_base_index" and summary.get("valid", True):
            calibrated_reference = summary

    for summary in summaries.values():
        summary["delta_vs_calibrated"] = _metric_delta(summary, calibrated_reference)

    manifest = _dataset_manifest(data_path)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "dataset_name": dataset_name,
        "data": str(data_path),
        "qa": str(qa_path),
        "index": str(index_path),
        "enriched_index": str(enriched_index_path),
        "top_k": top_k,
        "limit": limit,
        "strict_failed_evidence": strict_failed_evidence,
        "config": asdict(config),
        "dataset_manifest": manifest,
        "summary": summaries,
    }
    write_json(output_path / "summary.json", result)
    write_json(output_path / "details.json", details)
    _write_ablation_csv(output_path / "summary.csv", dataset_name, summaries)
    return result


def tune_retrieval(
    index_path: Path,
    dev_qa_path: Path,
    output_config_path: Path,
    top_k: int = 10,
    limit: int | None = None,
) -> dict[str, Any]:
    """Try predefined retrieval configs on a dev set and persist the best one."""
    candidates = [
        RetrievalConfig(use_dense=False, use_reranker=False, use_graph=False, use_enrichment=False),
        RetrievalConfig(use_dense=True, use_reranker=False, use_graph=False, use_enrichment=False),
        RetrievalConfig(use_dense=True, use_reranker=True, use_graph=False, use_enrichment=False),
        RetrievalConfig(use_dense=True, use_reranker=True, use_graph=False, use_enrichment=True),
        RetrievalConfig(use_dense=True, use_reranker=True, use_graph=True, use_enrichment=True),
    ]
    enrichment_010 = RetrievalConfig(use_dense=True, use_reranker=True, use_graph=False, use_enrichment=True)
    enrichment_010.weights["enrichment"] = 0.10
    candidates.append(enrichment_010)
    trials = []
    for config in candidates:
        result = evaluate_retrieval(
            index_path,
            dev_qa_path,
            top_k=top_k,
            mode="two_stage",
            calibrated=True,
            limit=limit,
            retrieval_config=config,
        )
        trials.append({"config": asdict(config), "metrics": {key: result[key] for key in SUMMARY_KEYS}})
    best = max(
        trials,
        key=lambda trial: (
            trial["metrics"]["all_evidence_success_at_k"],
            trial["metrics"]["recall_at_k"],
            trial["metrics"]["mrr"],
            -trial["metrics"]["latency_ms_p95"],
        ),
    )
    write_json(output_config_path, best["config"])
    write_json(output_config_path.with_name(output_config_path.stem + "_trials.json"), {"trials": trials})
    return best


def load_retrieval_config(path: Path | None) -> RetrievalConfig:
    """Load retrieval configuration or return defaults when no path is supplied."""
    if not path:
        return RetrievalConfig()
    return RetrievalConfig.from_path(path)


def _ablation_config(base: RetrievalConfig, signals: tuple[str, ...]) -> RetrievalConfig:
    """Clone a retrieval config and enable only the requested enrichment signals."""
    config = RetrievalConfig()
    values = asdict(base)
    for key, value in values.items():
        if key == "weights":
            config.weights = dict(value)
        elif hasattr(config, key):
            setattr(config, key, value)
    enabled = set(signals)
    config.use_enrichment = bool(enabled)
    config.use_generated_questions = "questions" in enabled
    config.use_semantic_summaries = "summaries" in enabled
    config.use_entity_enrichment = "entities" in enabled
    config.use_inferred_edges = "inferred_edges" in enabled
    config.use_edge_importance = "importance" in enabled
    config.use_contradiction_penalty = "contradiction_penalty" in enabled
    return config


def _metric_delta(
    summary: dict[str, Any],
    reference: dict[str, Any] | None,
) -> dict[str, float]:
    if not reference or not summary.get("valid", True):
        return {key: 0.0 for key in SUMMARY_KEYS if key not in {"count", "total_qa"}}
    return {
        key: round(float(summary.get(key, 0.0)) - float(reference.get(key, 0.0)), 6)
        for key in SUMMARY_KEYS
        if key not in {"count", "total_qa"}
    }


def _invalid_summary(
    name: str,
    index_path: Path,
    calibrated: bool,
    signals: tuple[str, ...],
    mode: str,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {key: 0.0 for key in SUMMARY_KEYS}
    summary.update({
        "index": str(index_path),
        "calibrated": calibrated,
        "retrieval_mode": mode,
        "signals": list(signals),
        "valid": False,
        "invalid_reason": "failed evidence chunks",
        "failed_evidence_units": len(failures),
        "variant": name,
    })
    return summary


def _dataset_manifest(data_path: Path) -> dict[str, Any]:
    manifest_path = data_path / "dataset_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _require_index(path: Path) -> None:
    required = ["documents.jsonl", "text_units.jsonl", "entities.jsonl", "calibrated_edges.jsonl"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Index {path} is missing required files: {', '.join(missing)}")


def _write_ablation_csv(path: Path, dataset_name: str, summaries: dict[str, dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "variant",
        "valid",
        "retrieval_mode",
        "signals",
        *SUMMARY_KEYS,
        "delta_vs_calibrated",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant, summary in summaries.items():
            row = {
                "dataset": dataset_name,
                "variant": variant,
                "valid": summary.get("valid", True),
                "retrieval_mode": summary.get("retrieval_mode", ""),
                "signals": "+".join(summary.get("signals", [])),
                "delta_vs_calibrated": json.dumps(summary.get("delta_vs_calibrated", {}), sort_keys=True),
            }
            row.update({key: summary.get(key, 0.0) for key in SUMMARY_KEYS})
            writer.writerow(row)
