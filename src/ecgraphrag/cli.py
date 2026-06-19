from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import load_retrieval_config, run_benchmark, run_retrieval_ablation, tune_retrieval
from .dataset import (
    create_qa_splits,
    download_multihop_rag_dataset,
    download_musique_ans_dataset,
    normalize_multihop_rag,
)
from .indexer import GraphRAGIndexer
from .metrics import compare_baseline_calibrated
from .retrieve import Retriever
from .visualize import export_graph


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level EC-GraphRAG command parser."""
    parser = argparse.ArgumentParser(prog="ecgraphrag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset")
    dataset.add_argument("--name", choices=["multihop_rag", "musique_ans"], default="multihop_rag")
    dataset.add_argument("--output", type=Path, default=Path("data/multihop_rag"))
    dataset.add_argument("--raw-dir", type=Path)
    dataset.add_argument("--limit-docs", type=int)
    dataset.add_argument("--limit-qa", type=int)

    index = subparsers.add_parser("index")
    index.add_argument("--input", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--chunk-size", type=int, default=600)
    index.add_argument("--overlap", type=int, default=100)
    index.add_argument("--extractor", choices=["rules", "llm"], default="rules")
    index.add_argument("--max-llm-units", type=int)
    index.add_argument("--no-resume", action="store_true")

    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--index", type=Path, required=True)
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--mode", choices=["heuristic", "embedding", "hybrid", "two_stage", "iterative"], default="two_stage")
    retrieve.add_argument("--weights", type=Path)
    retrieve.add_argument("--top-k", type=int, default=10)
    retrieve.add_argument("--max-hops", type=int, default=2)
    retrieve.add_argument("--token-budget", type=int, default=1200)
    retrieve.add_argument("--baseline", action="store_true", help="Use ordinary GraphRAG-style score without edge reliability")

    metrics = subparsers.add_parser("metrics")
    metrics.add_argument("--index", type=Path, required=True)
    metrics.add_argument("--qa", type=Path, required=True)
    metrics.add_argument("--top-k", type=int, default=10)
    metrics.add_argument("--mode", choices=["heuristic", "embedding", "hybrid", "two_stage", "iterative"], default="two_stage")
    metrics.add_argument("--limit", type=int)
    metrics.add_argument("--output", type=Path)
    metrics.add_argument("--config", type=Path)

    split = subparsers.add_parser("split")
    split.add_argument("--qa", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--index", type=Path, required=True)
    benchmark.add_argument("--enriched-index", type=Path)
    benchmark.add_argument("--qa", type=Path, required=True)
    benchmark.add_argument("--config", type=Path)
    benchmark.add_argument("--top-k", type=int, default=10)
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--output", type=Path)

    ablation = subparsers.add_parser("ablation")
    ablation.add_argument("--dataset-name", required=True)
    ablation.add_argument("--data", type=Path, required=True)
    ablation.add_argument("--index", type=Path, required=True)
    ablation.add_argument("--enriched-index", type=Path, required=True)
    ablation.add_argument("--output", type=Path, required=True)
    ablation.add_argument("--top-k", type=int, default=10)
    ablation.add_argument("--limit", type=int)
    ablation.add_argument("--config", type=Path)
    ablation.add_argument("--allow-failed-evidence", action="store_true")

    tune = subparsers.add_parser("tune-retrieval")
    tune.add_argument("--index", type=Path, required=True)
    tune.add_argument("--dev-qa", type=Path, required=True)
    tune.add_argument("--output-config", type=Path, required=True)
    tune.add_argument("--top-k", type=int, default=10)
    tune.add_argument("--limit", type=int)

    visualize = subparsers.add_parser("visualize")
    visualize.add_argument("--index", type=Path, required=True)
    visualize.add_argument("--output", type=Path)
    visualize.add_argument("--max-nodes", type=int, default=200)
    return parser


def main() -> None:
    """Dispatch the selected EC-GraphRAG CLI command."""
    args = build_parser().parse_args()
    if args.command == "dataset":
        if args.raw_dir:
            if args.name != "multihop_rag":
                raise ValueError("--raw-dir is only supported for --name multihop_rag")
            docs, qa = normalize_multihop_rag(args.raw_dir, args.limit_docs, args.limit_qa)
            from .storage import write_jsonl, write_json
            args.output.mkdir(parents=True, exist_ok=True)
            write_jsonl(args.output / "documents.jsonl", docs)
            write_jsonl(args.output / "qa.jsonl", qa)
            manifest = {"source": str(args.raw_dir), "documents": len(docs), "qa": len(qa)}
            write_json(args.output / "dataset_manifest.json", manifest)
        else:
            if args.name == "musique_ans":
                manifest = download_musique_ans_dataset(args.output, args.limit_qa)
            else:
                manifest = download_multihop_rag_dataset(args.output, args.limit_docs, args.limit_qa)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.command == "index":
        counts = GraphRAGIndexer(
            args.chunk_size,
            args.overlap,
            extractor=args.extractor,
            max_llm_units=args.max_llm_units,
            resume=not args.no_resume,
        ).index(args.input, args.output)
        print(json.dumps(counts, ensure_ascii=False, indent=2))
        return
    if args.command == "retrieve":
        result = Retriever(args.index, args.weights, calibrated=not args.baseline).retrieve(
            args.query, args.mode, args.top_k, args.max_hops, args.token_budget
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "metrics":
        result = compare_baseline_calibrated(
            args.index,
            args.qa,
            args.top_k,
            args.mode,
            args.limit,
            load_retrieval_config(args.config),
        )
        if args.output:
            from .storage import write_json
            write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return
    if args.command == "split":
        result = create_qa_splits(args.qa, args.output, args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "benchmark":
        result = run_benchmark(
            args.index,
            args.qa,
            output_path=args.output,
            enriched_index_path=args.enriched_index,
            config=load_retrieval_config(args.config),
            top_k=args.top_k,
            limit=args.limit,
        )
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return
    if args.command == "ablation":
        result = run_retrieval_ablation(
            args.dataset_name,
            args.data,
            args.index,
            args.enriched_index,
            args.output,
            config=load_retrieval_config(args.config),
            top_k=args.top_k,
            limit=args.limit,
            strict_failed_evidence=not args.allow_failed_evidence,
        )
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return
    if args.command == "tune-retrieval":
        result = tune_retrieval(
            args.index,
            args.dev_qa,
            args.output_config,
            top_k=args.top_k,
            limit=args.limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = export_graph(args.index, args.output, args.max_nodes)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
