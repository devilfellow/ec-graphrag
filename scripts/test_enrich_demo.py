"""Quick demo: run enrichment on 5 edges to verify API connectivity."""
import os
import sys
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
from ecgraphrag.models import Edge, Entity
from ecgraphrag.openrouter import OpenRouterClient
from ecgraphrag.storage import read_jsonl

INDEX_DIR = Path("work/multihop_graphrag_openrouter")

raw_edges = read_jsonl(INDEX_DIR / "calibrated_edges.jsonl")
raw_entities = read_jsonl(INDEX_DIR / "entities.jsonl")

edges = [Edge(**row) for row in raw_edges[:5]]
entities = [Entity(**row) for row in raw_entities[:5]]

print(f"Edges: {len(edges)}, Entities: {len(entities)}")
print(f"API key set: {bool(os.environ.get('OPENROUTER_API_KEY'))}")

client = OpenRouterClient()
print(f"Model: {client.config.model}")
print("Running enrich_graph with all steps on 5 edges...")

enriched_entities, enriched_edges = enrich_graph(
    entities, edges, client,
    steps=["questions", "summaries", "importance"]
)

print(f"\nResult: {len(enriched_edges)} edges")
for e in enriched_edges[:5]:
    print(f"\n--- {e.source} -> {e.target} ({e.relation}) ---")
    print(f"  questions: {e.generated_questions}")
    print(f"  summary: {e.semantic_summary}")
    print(f"  importance: {e.importance}")
