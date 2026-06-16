from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ecgraphrag.indexer import GraphRAGIndexer, _load_dotenv
from ecgraphrag.extract import _normalize_relation
from ecgraphrag.extract import extract_graph
from ecgraphrag.ingest import chunk_documents, ingest
from ecgraphrag.retrieve import Retriever
from ecgraphrag.storage import read_jsonl
from ecgraphrag.openrouter import OpenRouterConfig


class PipelineTest(unittest.TestCase):
    def test_relation_ontology_normalizes_unknown_relations(self) -> None:
        relation, ontology_ok = _normalize_relation("invented relationship")
        self.assertEqual(relation, "associated_with")
        self.assertFalse(ontology_ok)

    def test_llm_extraction_checkpoints_and_resumes(self) -> None:
        class PartialClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat_json(self, system: str, user: str, schema_hint: str | None = None) -> dict:
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("malformed response")
                return {
                    "entities": [{"name": "Alpha"}, {"name": "Beta"}],
                    "relationships": [{"source": "Alpha", "target": "Beta", "relation": "depends_on"}],
                }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.txt"
            source.write_text("Alpha depends on Beta. " * 20, encoding="utf-8")
            documents = ingest(source)
            units = chunk_documents(documents, chunk_size=12, overlap=2)
            cache = root / "index" / "llm_cache"
            stats: dict[str, int] = {}
            client = PartialClient()
            extract_graph(
                documents, units, extractor="llm", llm_client=client,
                llm_cache_dir=cache, continue_on_error=True, extraction_stats=stats,
            )
            self.assertEqual(stats["failed_units"], 1)
            self.assertTrue((cache.parent / "llm_errors.jsonl").exists())
            cached_before = len(list(cache.glob("*.json")))

            stats = {}
            second_client = PartialClient()
            second_client.calls = 2
            extract_graph(
                documents, units, extractor="llm", llm_client=second_client,
                llm_cache_dir=cache, resume=True, continue_on_error=True, extraction_stats=stats,
            )
            self.assertEqual(stats["cached_units"], cached_before)
            self.assertEqual(stats["failed_units"], 0)
            self.assertEqual((cache.parent / "llm_errors.jsonl").read_text().strip(), "")

    def test_llm_extraction_progress_is_not_printed_to_stdout(self) -> None:
        class Client:
            config = SimpleNamespace(workers=1)

            def chat_json(self, system: str, user: str, schema_hint: str | None = None) -> dict:
                return {
                    "entities": [{"name": "Alpha"}, {"name": "Beta"}],
                    "relationships": [{"source": "Alpha", "target": "Beta", "relation": "depends_on"}],
                }

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.txt"
            source.write_text("Alpha depends on Beta.", encoding="utf-8")
            documents = ingest(source)
            units = chunk_documents(documents, chunk_size=20, overlap=0)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                extract_graph(documents, units, extractor="llm", llm_client=Client())
            self.assertEqual(stdout.getvalue(), "")

    def test_llm_extraction_uses_configured_workers(self) -> None:
        class ConcurrentClient:
            def __init__(self) -> None:
                self.config = SimpleNamespace(workers=12)
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def chat_json(self, system: str, user: str, schema_hint: str | None = None) -> dict:
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.02)
                with self.lock:
                    self.active -= 1
                return {
                    "entities": [{"name": "Alpha"}, {"name": "Beta"}],
                    "relationships": [{"source": "Alpha", "target": "Beta", "relation": "depends_on"}],
                }

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.txt"
            source.write_text("Alpha depends on Beta. " * 80, encoding="utf-8")
            documents = ingest(source)
            units = chunk_documents(documents, chunk_size=8, overlap=0)
            client = ConcurrentClient()
            extract_graph(documents, units, extractor="llm", llm_client=client)
            self.assertGreater(client.max_active, 1)
            self.assertLessEqual(client.max_active, 12)

    def test_load_dotenv_overwrites_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            env_path = Path(temp) / ".env"
            env_path.write_text("OPENROUTER_MAX_TOKENS=10000\n", encoding="utf-8")
            with patch.dict(os.environ, {"OPENROUTER_MAX_TOKENS": "1800"}):
                _load_dotenv(env_path)
                self.assertEqual(os.environ["OPENROUTER_MAX_TOKENS"], "10000")

    def test_openrouter_workers_loaded_from_environment(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_WORKERS": "12"}):
            self.assertEqual(OpenRouterConfig.from_env().workers, 12)

    def test_llm_index_manifest_contains_resume_statistics(self) -> None:
        class Client:
            def __init__(self, config: OpenRouterConfig) -> None:
                self.config = config

            def chat_json(self, system: str, user: str, schema_hint: str | None = None) -> dict:
                return {
                    "entities": [{"name": "Alpha"}, {"name": "Beta"}],
                    "relationships": [{"source": "Alpha", "target": "Beta", "relation": "depends_on"}],
                }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.txt"
            source.write_text("Alpha depends on Beta.", encoding="utf-8")
            output = root / "index"
            config = OpenRouterConfig(api_key="test")
            with patch("ecgraphrag.indexer.OpenRouterConfig.from_env", return_value=config), patch(
                "ecgraphrag.indexer.OpenRouterClient", Client
            ):
                counts = GraphRAGIndexer(extractor="llm", resume=True).index(source, output)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(counts["successful_units"], 1)
            self.assertEqual(manifest["successful_units"], 1)
            self.assertEqual(manifest["failed_units"], 0)
            self.assertTrue(manifest["resume"])

    def test_index_calibrate_and_retrieve(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            output = root / "index"
            data.mkdir()
            rows = [
                {
                    "text": "Missing Wagons causes Equipment Downtime. Equipment Downtime causes Production Delay.",
                    "relationships": [
                        {"source": "Missing Wagons", "target": "Equipment Downtime", "relation": "causes"},
                        {"source": "Equipment Downtime", "target": "Production Delay", "relation": "causes"},
                    ],
                }
            ]
            (data / "facts.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )

            counts = GraphRAGIndexer(chunk_size=100, overlap=10).index(data, output)
            self.assertGreaterEqual(counts["relationships"], 2)
            edges = read_jsonl(output / "calibrated_edges.jsonl")
            self.assertTrue(all(0 <= edge["reliability"] <= 1 for edge in edges))
            self.assertTrue(all(edge["evidence_score"] > 0 for edge in edges))
            try:
                import pyarrow
            except ImportError:
                pass
            else:
                self.assertTrue((output / "relationships.parquet").exists())

            result = Retriever(output).retrieve(
                "Why is production delayed?", mode="heuristic", max_hops=2, token_budget=300
            )
            self.assertTrue(result["context"])
            self.assertLessEqual(result["tokens_used"], 300)
            self.assertTrue(any(item["kind"] in {"edge", "path"} for item in result["context"]))

    def test_hybrid_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.txt"
            source.write_text("Alpha System depends on Beta Service.", encoding="utf-8")
            output = root / "index"
            counts = GraphRAGIndexer(chunk_size=50, overlap=5).index(source, output)
            self.assertGreaterEqual(counts["relationships"], 1)
            result = Retriever(output).retrieve("What depends on Beta Service?", mode="hybrid")
            self.assertEqual(result["mode"], "hybrid")
            self.assertTrue(result["context"])


if __name__ == "__main__":
    unittest.main()
