"""Run enrichment, save an enriched index, and evaluate retrieval metrics."""
import os
import sys
import json
import shutil
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.chdir(Path(__file__).resolve().parent.parent)

env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from ecgraphrag.enrich import enrich_graph
from ecgraphrag.metrics import evaluate_retrieval
from ecgraphrag.models import Edge, Entity
from ecgraphrag.openrouter import OpenRouterClient
from ecgraphrag.storage import export_table, read_jsonl

INDEX_DIR = Path("work/multihop_graphrag_openrouter")
ENRICHED_DIR = Path("work/multihop_graphrag_enriched")
DATA_DIR = Path("data/multihop_rag")
QA_PATH = DATA_DIR / "qa.jsonl"

MAX_ENRICH_EDGES = 20
MAX_ENRICH_ENTITIES = 20
ENRICH_STEPS = ["questions", "summaries", "contradictions", "entities", "infer", "importance"]
TOP_K = 10
LIMIT_QA = 25

print("=== Step 1: Loading edges and entities ===")
raw_edges = read_jsonl(INDEX_DIR / "calibrated_edges.jsonl")
raw_entities = read_jsonl(INDEX_DIR / "entities.jsonl")
all_edges = [Edge(**row) for row in raw_edges]
all_entities = [Entity(**row) for row in raw_entities]
print(f"  Total: {len(all_edges)} edges, {len(all_entities)} entities")

edges_to_enrich = all_edges[:MAX_ENRICH_EDGES]
entities_to_enrich = all_entities[:MAX_ENRICH_ENTITIES]
print(f"  Enriching: {len(edges_to_enrich)} edges, {len(entities_to_enrich)} entities")
print(f"  Steps: {ENRICH_STEPS}")

print("\n=== Step 2: LLM Enrichment (this may take a few minutes) ===")
client = OpenRouterClient()
print(f"  Model: {client.config.model}")

enriched_entities_subset, enriched_edges_subset = enrich_graph(
    entities_to_enrich, edges_to_enrich, client, steps=ENRICH_STEPS
)

enriched_edge_ids = {e.id for e in enriched_edges_subset}
remaining_edges = [e for e in all_edges if e.id not in enriched_edge_ids]
enriched_edges = enriched_edges_subset + remaining_edges

enriched_entity_ids = {e.id for e in enriched_entities_subset}
remaining_entities = [e for e in all_entities if e.id not in enriched_entity_ids]
enriched_entities = enriched_entities_subset + remaining_entities

new_inferred = len(enriched_edges_subset) - len(edges_to_enrich)
print(f"  Result: {len(enriched_edges)} edges (was {len(all_edges)}), +{new_inferred} inferred")

print("\n=== Step 3: Saving enriched index ===")
if ENRICHED_DIR.exists():
    shutil.rmtree(ENRICHED_DIR)
shutil.copytree(INDEX_DIR, ENRICHED_DIR)
export_table(ENRICHED_DIR, "calibrated_edges", [asdict(e) for e in enriched_edges])
export_table(ENRICHED_DIR, "entities", [asdict(e) for e in enriched_entities])
print(f"  Saved to: {ENRICHED_DIR}")

print("\n=== Step 4: Evaluating retrieval (3 variants) ===")
baseline = evaluate_retrieval(INDEX_DIR, QA_PATH, top_k=TOP_K, mode="two_stage", calibrated=False, limit=LIMIT_QA)
calibrated = evaluate_retrieval(INDEX_DIR, QA_PATH, top_k=TOP_K, mode="two_stage", calibrated=True, limit=LIMIT_QA)
enriched_eval = evaluate_retrieval(ENRICHED_DIR, QA_PATH, top_k=TOP_K, mode="two_stage", calibrated=True, limit=LIMIT_QA)

metrics_cols = [
    "all_evidence_success_at_k", "recall_at_k", "precision_at_k",
    "mrr", "ndcg_at_k", "packed_context_recall_at_k", "answer_hit_rate",
]

print("\n=== RESULTS ===")
print(f"{'Variant':<25} {'Success@K':>10} {'Recall@K':>10} {'Prec@K':>10} {'MRR':>10} {'nDCG@K':>10}")
print("-" * 79)
for name, res in [("baseline", baseline), ("calibrated", calibrated), ("enriched+calibrated", enriched_eval)]:
    print(
        f"{name:<25} {res['all_evidence_success_at_k']:>10.4f} {res['recall_at_k']:>10.4f} "
        f"{res['precision_at_k']:>10.4f} {res['mrr']:>10.4f} {res['ndcg_at_k']:>10.4f}"
    )

print(f"\n{'Delta':<25} {'Success@K':>10} {'Recall@K':>10} {'Prec@K':>10} {'MRR':>10} {'nDCG@K':>10}")
print("-" * 79)
delta_b = {k: enriched_eval[k] - baseline[k] for k in metrics_cols}
delta_c = {k: enriched_eval[k] - calibrated[k] for k in metrics_cols}
for name, delta in [("enriched - baseline", delta_b), ("enriched - calibrated", delta_c)]:
    print(
        f"{name:<25} {delta['all_evidence_success_at_k']:>+10.4f} {delta['recall_at_k']:>+10.4f} "
        f"{delta['precision_at_k']:>+10.4f} {delta['mrr']:>+10.4f} {delta['ndcg_at_k']:>+10.4f}"
    )

full_result = {
    "summary": {
        "baseline": {k: baseline[k] for k in metrics_cols + ["count", "mode"]},
        "calibrated": {k: calibrated[k] for k in metrics_cols + ["count", "mode"]},
        "enriched": {k: enriched_eval[k] for k in metrics_cols + ["count", "mode"]},
    },
    "delta_enriched_vs_baseline": {k: round(v, 6) for k, v in delta_b.items()},
    "delta_enriched_vs_calibrated": {k: round(v, 6) for k, v in delta_c.items()},
    "enrich_steps": ENRICH_STEPS,
    "enriched_edge_count": len(enriched_edges),
    "original_edge_count": len(all_edges),
    "max_enrich_edges": MAX_ENRICH_EDGES,
}
metrics_path = ENRICHED_DIR / "metrics_baseline_vs_calibrated_vs_enriched.json"
metrics_path.write_text(json.dumps(full_result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nMetrics saved: {metrics_path}")

print("\n=== Enrichment Statistics ===")
enriched_with_q = sum(1 for e in enriched_edges if e.generated_questions)
enriched_with_s = sum(1 for e in enriched_edges if e.semantic_summary)
enriched_with_c = sum(1 for e in enriched_edges if e.contradiction_info)
inferred_count = sum(1 for e in enriched_edges if e.evidence_type == "inferred")
high_importance = sum(1 for e in enriched_edges if e.importance > 0.7)
low_importance = sum(1 for e in enriched_edges if e.importance < 0.3)
print(f"  With generated_questions: {enriched_with_q}/{len(enriched_edges)}")
print(f"  With semantic_summary:    {enriched_with_s}/{len(enriched_edges)}")
print(f"  With contradiction_info:  {enriched_with_c}/{len(enriched_edges)}")
print(f"  Inferred edges:           {inferred_count}")
print(f"  Importance > 0.7:         {high_importance}")
print(f"  Importance < 0.3:         {low_importance}")
