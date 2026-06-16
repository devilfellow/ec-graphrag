from __future__ import annotations

import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import evaluate_retrieval
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
