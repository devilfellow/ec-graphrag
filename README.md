# EC-GraphRAG

Evidence-calibrated GraphRAG pipeline for MultiHop-RAG: the project builds a knowledge graph from documents, calibrates edge reliability, optionally enriches graph artifacts with LLM calls, and evaluates document retrieval against gold evidence.

## Limitations And Safety

- LLM extraction and enrichment use OpenRouter and can incur API costs. Resume/cache is enabled for successful chunks, but uncached chunks are submitted to the model.
- Keep `OPENROUTER_API_KEY` only in a local `.env`. Do not commit API keys or notebook outputs containing secrets.
- The package requires Python `>=3.11`. For notebook work from Windows, prefer a WSL-based virtual environment and Jupyter kernel.
- Retrieval metrics are evidence-document metrics, not generated-answer quality metrics.
- HTML/PNG graph exports are auxiliary. Auditable outputs are stored as JSONL, Parquet, GraphML, metrics JSON, and manifest files.
- If embedding or reranker models are unavailable, retrieval can fall back to deterministic lexical/hashed behavior unless strict model mode is enabled.

## Documentation

- [Full project workflow and metric description](docs/PROJECT_DESCRIPTION.md)
- [Default retrieval configuration](configs/retrieval_two_stage.json)
- [Evaluation notebook](notebooks/evaluate_graphrag_vs_calibrated.ipynb)

Background references:

- [GraphRAG documentation](https://microsoft.github.io/graphrag/)
- [GraphRAG paper](https://arxiv.org/abs/2404.16130)
- [MultiHop-RAG dataset](https://github.com/yixuantt/MultiHop-RAG/)
- [OpenRouter API](https://openrouter.ai/docs/quickstart)
- [Sentence Transformers](https://sbert.net/)
- [NetworkX](https://networkx.org/documentation/stable/)

## Architecture

```text
documents
  -> text units
  -> graph extraction
  -> edge calibration
  -> communities and reports
  -> optional LLM enrichment
  -> two-stage retrieval
  -> retrieval metrics
```

Main components:

- `ecgraphrag.dataset`: downloads and normalizes MultiHop-RAG.
- `ecgraphrag.ingest`: loads documents and creates overlapping text units.
- `ecgraphrag.extract`: extracts entities and relationships with rules or LLM.
- `ecgraphrag.calibrate`: computes reliability scores for graph edges.
- `ecgraphrag.enrich`: adds generated questions, summaries, entity descriptions, inferred edges, contradiction analysis, and importance.
- `ecgraphrag.retrieve`: ranks documents and packs retrieval context.
- `ecgraphrag.metrics`: evaluates baseline, calibrated, and enriched retrieval.
- `ecgraphrag.visualize`: exports GraphML, HTML, PNG, and export manifest.

## Installation

Bash/WSL:

```bash
git clone https://github.com/devilfellow/ec-graphrag.git
cd ec-graphrag
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[visual,notebook,dev]"
```

PowerShell:

```powershell
git clone https://github.com/devilfellow/ec-graphrag.git
cd ec-graphrag
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[visual,notebook,dev]"
```

Create `.env` in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=qwen/qwen3.7-plus
OPENROUTER_TEMPERATURE=0
OPENROUTER_MAX_TOKENS=10000
OPENROUTER_TIMEOUT=90
OPENROUTER_RETRIES=3
OPENROUTER_WORKERS=12
```

After changing `.env` in a running notebook, rerun the configuration cell or restart the kernel.

## Quickstart

Run a small MultiHop-RAG smoke experiment:

```bash
python -m ecgraphrag dataset \
  --raw-dir data/multihop_rag/raw \
  --output data/multihop_rag \
  --limit-docs 50 \
  --limit-qa 25

python -m ecgraphrag index \
  --input data/multihop_rag/documents.jsonl \
  --output work/multihop_graphrag_openrouter \
  --extractor llm \
  --chunk-size 600 \
  --overlap 100

python -m ecgraphrag retrieve \
  --index work/multihop_graphrag_openrouter \
  --query "Which articles are related to the same event?" \
  --mode two_stage \
  --top-k 10

python -m ecgraphrag benchmark \
  --index work/multihop_graphrag_openrouter \
  --qa data/multihop_rag/qa.jsonl \
  --top-k 10 \
  --limit 25 \
  --output work/multihop_graphrag_openrouter/smoke_benchmark.json
```

If raw MultiHop-RAG files are not available locally, omit `--raw-dir`; the dataset command will download them into `data/multihop_rag/raw`.

## Common Commands

Prepare a dataset:

```bash
python -m ecgraphrag dataset \
  --output data/multihop_rag \
  --limit-docs 50 \
  --limit-qa 25
```

Build an index without LLM extraction:

```bash
python -m ecgraphrag index \
  --input data/multihop_rag/documents.jsonl \
  --output work/multihop_rules_index \
  --extractor rules
```

Build an index with LLM extraction:

```bash
python -m ecgraphrag index \
  --input data/multihop_rag/documents.jsonl \
  --output work/multihop_graphrag_openrouter \
  --extractor llm
```

LLM extraction resumes by default. Successful responses are stored in `llm_cache/<text_unit_id>.json`, unresolved failures are stored in `llm_errors.jsonl`, and OpenRouter diagnostics are stored in `llm_cache/openrouter_diagnostics.jsonl`.

Run retrieval:

```bash
python -m ecgraphrag retrieve \
  --index work/multihop_graphrag_openrouter \
  --query "Which articles are related to the same event?" \
  --mode two_stage \
  --top-k 10
```

Evaluate baseline and calibrated retrieval:

```bash
python -m ecgraphrag metrics \
  --index work/multihop_graphrag_openrouter \
  --qa data/multihop_rag/qa.jsonl \
  --mode two_stage \
  --top-k 10 \
  --limit 25 \
  --output work/multihop_graphrag_openrouter/metrics.json
```

Run a full train/dev/test workflow:

```bash
python -m ecgraphrag dataset \
  --raw-dir data/multihop_rag/raw \
  --output data/multihop_rag_full

python -m ecgraphrag split \
  --qa data/multihop_rag_full/qa.jsonl \
  --output data/multihop_rag_full/splits

python -m ecgraphrag index \
  --input data/multihop_rag_full/documents.jsonl \
  --output work/multihop_full_index \
  --extractor llm

python -m ecgraphrag tune-retrieval \
  --index work/multihop_full_index \
  --dev-qa data/multihop_rag_full/splits/dev.jsonl \
  --output-config work/retrieval_config.json

python -m ecgraphrag benchmark \
  --index work/multihop_full_index \
  --qa data/multihop_rag_full/splits/test.jsonl \
  --config work/retrieval_config.json \
  --output work/retrieval_benchmark.json
```

## Enrichment

The enrichment pipeline adds retrieval-oriented graph fields: generated questions, semantic summaries, contradiction information, enriched entity descriptions, inferred edges, and importance scores.

```bash
python scripts/run_enrichment_pipeline.py
```

The default enriched index location used by scripts and notebook cells is:

```text
work/multihop_graphrag_enriched/
```

## Visualization

```bash
python -m ecgraphrag visualize \
  --index work/multihop_graphrag_enriched \
  --output work/multihop_graphrag_enriched/graph_exports \
  --max-nodes 200
```

Expected outputs:

```text
graph.graphml
enriched_graph.html
graph.png
graph_export_manifest.json
```

If HTML export fails, inspect `graph_export_manifest.json` for `html_error`.

## Index Outputs

```text
work/<index>/
  documents.jsonl / documents.parquet
  text_units.jsonl / text_units.parquet
  entities.jsonl / entities.parquet
  relationships.jsonl / relationships.parquet
  calibrated_edges.jsonl / calibrated_edges.parquet
  communities.jsonl / communities.parquet
  community_reports.jsonl / community_reports.parquet
  manifest.json
  llm_cache/
  llm_errors.jsonl
```

## Tests

```bash
python -m unittest discover -s tests -v
```

For low-cost validation, use `--extractor rules` or a small `--max-llm-units` value before running full LLM indexing.
