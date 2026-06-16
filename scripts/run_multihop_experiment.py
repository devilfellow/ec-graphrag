from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecgraphrag.dataset import download_multihop_rag_dataset
from ecgraphrag.indexer import GraphRAGIndexer
from ecgraphrag.metrics import compare_baseline_calibrated
from ecgraphrag.visualize import export_graph


def main() -> None:
    """Run a small end-to-end MultiHop-RAG experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/multihop_rag"))
    parser.add_argument("--index", type=Path, default=Path("work/multihop_index"))
    parser.add_argument("--limit-docs", type=int, default=50)
    parser.add_argument("--limit-qa", type=int, default=25)
    parser.add_argument("--extractor", choices=["rules", "llm"], default="llm")
    parser.add_argument("--max-llm-units", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    manifest = download_multihop_rag_dataset(args.data, args.limit_docs, args.limit_qa)
    counts = GraphRAGIndexer(extractor=args.extractor, max_llm_units=args.max_llm_units).index(
        args.data / "documents.jsonl", args.index
    )
    metrics = compare_baseline_calibrated(
        args.index, args.data / "qa.jsonl", top_k=args.top_k, mode="two_stage", limit=args.limit_qa
    )
    graph = export_graph(args.index, args.index / "graph_exports", max_nodes=120)
    result = {"dataset": manifest, "counts": counts, "metrics": metrics["summary"], "graph": graph}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
