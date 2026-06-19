from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from ecgraphrag.benchmark import ABLATION_VARIANTS, run_retrieval_ablation
from ecgraphrag.models import CommunityReport, Document, Edge, Entity, TextUnit
from ecgraphrag.retrieve import RetrievalConfig
from ecgraphrag.storage import export_table, write_json, write_jsonl


class AblationTest(unittest.TestCase):
    def _write_index(self, path: Path, enriched: bool) -> None:
        path.mkdir()
        documents = [
            Document("doc_gold", "Gold", "Alpha causes Beta.", "source", {"id": "gold-url", "title": "Gold"}),
            Document("doc_other", "Other", "Unrelated weather document.", "source", {"id": "other-url", "title": "Other"}),
        ]
        units = [
            TextUnit("tu_gold", "doc_gold", "Alpha causes Beta.", 0, 4),
            TextUnit("tu_other", "doc_other", "Unrelated weather document.", 0, 4),
        ]
        edge = Edge(
            "edge_gold", "Alpha", "Beta", "causes", "Alpha causes Beta",
            source_docs=["doc_gold"], text_unit_ids=["tu_gold"],
            evidence_text="Alpha causes Beta.", edge_text="Alpha causes Beta.",
            reliability=0.9,
        )
        if enriched:
            edge.generated_questions = ["What does Alpha cause?"]
            edge.semantic_summary = "Alpha creates the Beta result."
            edge.importance = 0.9
        edges = [edge]
        if enriched:
            edges.append(Edge(
                "edge_inferred", "Alpha", "Gamma", "associated_with", "Alpha implies Gamma",
                source_docs=["doc_gold"], text_unit_ids=["tu_gold"],
                evidence_text="Inferred from Alpha causes Beta.", edge_text="Alpha inferred relation to Gamma.",
                evidence_type="inferred", reliability=0.5,
            ))
        entities = [Entity("alpha", "Alpha", enriched_description="Alpha enriched entity"), Entity("beta", "Beta")]
        reports = [CommunityReport("r1", "c1", "Report", "Alpha Beta", [], ["edge_gold"], 0.9)]
        for name, values in (
            ("documents", documents),
            ("text_units", units),
            ("entities", entities),
            ("relationships", edges),
            ("calibrated_edges", edges),
            ("community_reports", reports),
        ):
            export_table(path, name, [asdict(value) for value in values])

    def test_ablation_outputs_all_variants_without_enrichment_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            write_jsonl(data / "qa.jsonl", [{
                "id": "q1",
                "query": "What does Alpha cause?",
                "answer": "Beta",
                "evidence": [{"url": "gold-url", "title": "Gold"}],
            }])
            write_json(data / "dataset_manifest.json", {"documents": 2, "qa": 1})
            base = root / "base"
            enriched = root / "enriched"
            output = root / "ablation"
            self._write_index(base, enriched=False)
            self._write_index(enriched, enriched=True)

            with patch("ecgraphrag.enrich.enrich_graph", side_effect=AssertionError):
                result = run_retrieval_ablation(
                    "test_dataset",
                    data,
                    base,
                    enriched,
                    output,
                    config=RetrievalConfig(use_dense=False, use_reranker=False),
                    top_k=2,
                )

            self.assertEqual(set(result["summary"]), {name for name, *_ in ABLATION_VARIANTS})
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "details.json").exists())
            with (output / "summary.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(ABLATION_VARIANTS))
            self.assertIn("delta_vs_calibrated", rows[0])
            saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("delta_vs_calibrated", saved["summary"]["all_enrichment_fields"])

    def test_ablation_can_warn_instead_of_invalidating_failed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            write_jsonl(data / "qa.jsonl", [{
                "id": "q1",
                "query": "What does Alpha cause?",
                "answer": "Beta",
                "evidence": [{"url": "gold-url", "title": "Gold"}],
            }])
            base = root / "base"
            enriched = root / "enriched"
            output = root / "ablation"
            self._write_index(base, enriched=False)
            self._write_index(enriched, enriched=True)
            write_jsonl(base / "llm_errors.jsonl", [{
                "text_unit_id": "tu_gold",
                "document_id": "doc_gold",
                "error": "failed",
            }])
            write_jsonl(enriched / "llm_errors.jsonl", [{
                "text_unit_id": "tu_gold",
                "document_id": "doc_gold",
                "error": "failed",
            }])

            result = run_retrieval_ablation(
                "test_dataset",
                data,
                base,
                enriched,
                output,
                config=RetrievalConfig(use_dense=False, use_reranker=False),
                top_k=2,
                strict_failed_evidence=False,
            )

            row = result["summary"]["calibrated_base_index"]
            self.assertTrue(row["valid"])
            self.assertEqual(row["failed_evidence_units"], 1)
            self.assertEqual(row["warning"], "failed evidence chunks present")
            self.assertGreater(row["count"], 0)


if __name__ == "__main__":
    unittest.main()
