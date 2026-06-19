from __future__ import annotations

import tempfile
import sys
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from ecgraphrag.metrics import evaluate_retrieval
from ecgraphrag.models import CommunityReport, Document, Edge, Entity, TextUnit
from ecgraphrag.retrieve import RetrievalConfig, Retriever, decompose_query
from ecgraphrag.storage import export_table, write_jsonl
from ecgraphrag import text


class EvidenceMetricsTest(unittest.TestCase):
    def _index(self, root: Path, include_inferred: bool = False) -> Path:
        index = root / "index"
        index.mkdir()
        documents = [
            Document("doc_gold", "Gold", "Alpha causes Beta.", "source", {"id": "gold-url", "title": "Gold"}),
            Document("doc_other", "Other", "Weather is sunny.", "source", {"id": "other-url", "title": "Other"}),
        ]
        units = [
            TextUnit("tu_gold", "doc_gold", "Alpha causes Beta.", 0, 4),
            TextUnit("tu_other", "doc_other", "Weather is sunny.", 0, 4),
        ]
        edge = Edge(
            "edge_gold", "Alpha", "Beta", "causes", "Alpha causes Beta",
            source_docs=["doc_gold"], text_unit_ids=["tu_gold"],
            evidence_text="Alpha causes Beta.", edge_text="Alpha causes Beta.",
            reliability=0.9,
            generated_questions=["Which component produces the target result?"],
            semantic_summary="Alpha production creates the Beta target result.",
            importance=0.9,
        )
        edges = [edge]
        if include_inferred:
            edges.append(Edge(
                "edge_inferred", "Gamma", "Delta", "associated_with", "Gamma implies Delta",
                source_docs=["doc_other"], text_unit_ids=["tu_other"],
                evidence_text="Inferred from graph path.", edge_text="Gamma inferred bridge to Delta.",
                reliability=0.5, evidence_type="inferred",
            ))
        entities = [Entity("a", "Alpha", aliases=["A"]), Entity("b", "Beta")]
        entities[0].enriched_description = "Alpha is an enriched entity description."
        entities[0].category = "Component"
        reports = [
            CommunityReport(f"r{i}", f"c{i}", f"Report {i}", "Alpha Beta", [], ["edge_gold"], 0.9)
            for i in range(6)
        ]
        for name, values in (
            ("documents", documents),
            ("text_units", units),
            ("entities", entities),
            ("relationships", edges),
            ("calibrated_edges", edges),
            ("community_reports", reports),
        ):
            export_table(index, name, [asdict(value) for value in values])
        return index

    def test_metrics_require_complete_evidence_and_use_document_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = self._index(root)
            qa = root / "qa.jsonl"
            write_jsonl(qa, [
                {
                    "id": "covered",
                    "query": "What does Alpha cause?",
                    "answer": "Beta",
                    "evidence": [{"url": "gold-url", "title": "Gold"}],
                },
                {
                    "id": "missing",
                    "query": "What is missing?",
                    "answer": "Missing",
                    "evidence": [{"url": "missing-url", "title": "Missing"}],
                },
            ])
            result = evaluate_retrieval(index, qa, mode="hybrid")
            self.assertEqual(result["count"], 1)
            self.assertEqual(result["skipped_without_evidence"], 1)
            self.assertGreater(result["mrr"], 0)
            self.assertEqual(result["all_evidence_success_at_k"], 1.0)
            self.assertGreater(result["ndcg_at_k"], 0)

    def test_retrieval_limits_reports_and_exposes_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            retriever = Retriever(self._index(Path(temp)))
            contexts = retriever.retrieve("What does A cause?", mode="hybrid", top_k=10)["context"]
            self.assertLessEqual(sum(item["kind"] == "report" for item in contexts), 2)
            self.assertTrue(any("doc_gold" in item["metadata"]["document_ids"] for item in contexts))

    def test_two_stage_ranking_is_independent_from_context_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            retriever = Retriever(
                self._index(Path(temp)),
                config=RetrievalConfig(use_dense=False, use_reranker=False),
            )
            result = retriever.retrieve(
                "What does Alpha cause?",
                mode="two_stage",
                top_k=2,
                token_budget=1,
            )
            self.assertEqual(result["ranked_documents"][0]["id"], "doc_gold")
            self.assertEqual(result["context"], [])

    def test_iterative_retrieval_uses_bridge_entities(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = root / "index"
            index.mkdir()
            documents = [
                Document("doc_seed", "Green album", "Green was performed by Steve Hillage.", "source", {"id": "seed"}),
                Document("doc_bridge", "Miquette Giraudy", "Miquette Giraudy is the spouse and partner of Steve Hillage.", "source", {"id": "bridge"}),
                Document("doc_noise", "Green color", "Green is a color with many cultural associations.", "source", {"id": "noise"}),
            ]
            units = [
                TextUnit("tu_seed", "doc_seed", documents[0].text, 0, 8),
                TextUnit("tu_bridge", "doc_bridge", documents[1].text, 0, 10),
                TextUnit("tu_noise", "doc_noise", documents[2].text, 0, 9),
            ]
            edges = [
                Edge("e_seed", "Green", "Steve Hillage", "associated_with", "Green performer",
                     source_docs=["doc_seed"], text_unit_ids=["tu_seed"], edge_text="Green performer Steve Hillage.", reliability=0.9),
                Edge("e_bridge", "Miquette Giraudy", "Steve Hillage", "associated_with", "Spouse",
                     source_docs=["doc_bridge"], text_unit_ids=["tu_bridge"], edge_text="Miquette Giraudy spouse Steve Hillage.", reliability=0.9),
                Edge("e_noise", "Green", "Color", "associated_with", "Green color",
                     source_docs=["doc_noise"], text_unit_ids=["tu_noise"], edge_text="Green color.", reliability=0.6),
            ]
            entities = [Entity("green", "Green"), Entity("steve", "Steve Hillage"), Entity("miquette", "Miquette Giraudy")]
            reports = [CommunityReport("r1", "c1", "Report", "Green Steve Hillage", [], ["e_seed", "e_bridge"], 0.9)]
            for name, values in (
                ("documents", documents),
                ("text_units", units),
                ("entities", entities),
                ("relationships", edges),
                ("calibrated_edges", edges),
                ("community_reports", reports),
            ):
                export_table(index, name, [asdict(value) for value in values])

            retriever = Retriever(index, config=RetrievalConfig(use_dense=False, use_reranker=False, use_enrichment=False))
            result = retriever.retrieve("Who is the spouse of the Green performer?", mode="iterative", top_k=2)
            ranked_ids = [row["id"] for row in result["ranked_documents"]]
            self.assertIn("doc_seed", ranked_ids)
            self.assertIn("doc_bridge", ranked_ids)
            self.assertIn("Steve Hillage", result["diagnostics"]["bridge_entities"])

    def test_two_stage_ignores_graph_enrichment_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            retriever = Retriever(
                self._index(Path(temp)),
                config=RetrievalConfig(
                    use_dense=False, use_reranker=False, use_graph=False, use_enrichment=False
                ),
            )
            with patch.object(retriever, "_document_graph_text", side_effect=AssertionError):
                ranked, diagnostics = retriever.rank_documents("What does Alpha cause?", top_k=2)
            self.assertEqual(ranked[0]["id"], "doc_gold")
            self.assertNotIn("graph", diagnostics)

    def test_enrichment_is_scored_as_separate_document_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            retriever = Retriever(
                self._index(Path(temp)),
                config=RetrievalConfig(
                    use_dense=False, use_reranker=False, use_graph=False, use_enrichment=True
                ),
            )
            retriever.edges[0].generated_questions = ["Which component produces the target result?"]
            ranked, diagnostics = retriever.rank_documents("Which component produces the target result?", top_k=2)
            self.assertGreater(ranked[0]["features"]["enrichment"], 0)
            self.assertGreater(diagnostics["ranked_lists"], 4)

    def test_granular_enrichment_flags_select_document_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            index = self._index(Path(temp), include_inferred=True)
            cases = [
                ("questions", RetrievalConfig(use_dense=False, use_reranker=False, use_generated_questions=True,
                                              use_semantic_summaries=False, use_entity_enrichment=False,
                                              use_inferred_edges=False, use_edge_importance=False,
                                              use_contradiction_penalty=False)),
                ("summaries", RetrievalConfig(use_dense=False, use_reranker=False, use_generated_questions=False,
                                              use_semantic_summaries=True, use_entity_enrichment=False,
                                              use_inferred_edges=False, use_edge_importance=False,
                                              use_contradiction_penalty=False)),
                ("entities", RetrievalConfig(use_dense=False, use_reranker=False, use_generated_questions=False,
                                             use_semantic_summaries=False, use_entity_enrichment=True,
                                             use_inferred_edges=False, use_edge_importance=False,
                                             use_contradiction_penalty=False)),
                ("inferred_edges", RetrievalConfig(use_dense=False, use_reranker=False, use_generated_questions=False,
                                                   use_semantic_summaries=False, use_entity_enrichment=False,
                                                   use_inferred_edges=True, use_edge_importance=False,
                                                   use_contradiction_penalty=False)),
            ]
            for expected, config in cases:
                retriever = Retriever(index, config=config)
                _, diagnostics = retriever.rank_documents("target result", top_k=2)
                self.assertEqual(diagnostics["enrichment_fields"], [expected])

    def test_use_enrichment_false_disables_all_granular_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = RetrievalConfig(
                use_dense=False, use_reranker=False, use_enrichment=False,
                use_generated_questions=True, use_semantic_summaries=True,
                use_entity_enrichment=True, use_inferred_edges=True,
                use_edge_importance=True, use_contradiction_penalty=True,
            )
            retriever = Retriever(self._index(Path(temp), include_inferred=True), config=config)
            _, diagnostics = retriever.rank_documents("Which component produces the target result?", top_k=2)
            self.assertEqual(diagnostics["enrichment_fields"], [])
            self.assertTrue(all(edge.evidence_type != "inferred" for edge in retriever.edges))

    def test_importance_and_contradiction_features_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            index = self._index(Path(temp))
            config = RetrievalConfig(
                use_dense=False, use_reranker=False, use_generated_questions=False,
                use_semantic_summaries=False, use_entity_enrichment=False,
                use_inferred_edges=False, use_edge_importance=True,
                use_contradiction_penalty=True,
            )
            retriever = Retriever(index, config=config)
            retriever.edges[0].conflict_status = "contradiction"
            retriever.document_edges["doc_gold"][0].conflict_status = "contradiction"
            ranked, diagnostics = retriever.rank_documents("What does Alpha cause?", top_k=2)
            self.assertEqual(diagnostics["enrichment_fields"], [])
            self.assertGreater(ranked[0]["features"]["importance"], 0)
            self.assertLess(ranked[0]["features"]["contradiction_penalty"], 0)

    def test_query_decomposition_keeps_full_query_and_clauses(self) -> None:
        query = "Which company was reported by Fortune and which product was described by TechCrunch?"
        parts = decompose_query(query, max_subqueries=4)
        self.assertEqual(parts[0], query)
        self.assertGreaterEqual(len(parts), 2)

    def test_query_decomposition_does_not_duplicate_unsplit_question(self) -> None:
        query = "Who is the spouse of the Green performer?"
        self.assertEqual(decompose_query(query), [query])

    def test_semantic_similarity_falls_back_when_transformers_rejects_keras(self) -> None:
        module = types.ModuleType("sentence_transformers")

        class IncompatibleSentenceTransformer:
            def __init__(self, model_name: str) -> None:
                raise ValueError("Keras 3 is not yet supported in Transformers")

        module.SentenceTransformer = IncompatibleSentenceTransformer
        previous_model = text._SEMANTIC_MODEL
        previous_model_name = text._SEMANTIC_MODEL_NAME
        previous_unavailable = text._SEMANTIC_MODEL_UNAVAILABLE
        try:
            text._SEMANTIC_MODEL = None
            text._SEMANTIC_MODEL_UNAVAILABLE = False
            with patch.dict(sys.modules, {"sentence_transformers": module}):
                scores = text.semantic_similarity_scores("Alpha", ["Alpha", "Beta"])
            self.assertEqual(len(scores), 2)
            self.assertGreater(scores[0], scores[1])
            self.assertTrue(text._SEMANTIC_MODEL_UNAVAILABLE)
        finally:
            text._SEMANTIC_MODEL = previous_model
            text._SEMANTIC_MODEL_NAME = previous_model_name
            text._SEMANTIC_MODEL_UNAVAILABLE = previous_unavailable

    def test_strict_semantic_similarity_rejects_fallback(self) -> None:
        previous_unavailable = text._SEMANTIC_MODEL_UNAVAILABLE
        try:
            text._SEMANTIC_MODEL_UNAVAILABLE = True
            with self.assertRaises(RuntimeError):
                text.semantic_similarity_scores("Alpha", ["Beta"], model_name="required", strict=True)
        finally:
            text._SEMANTIC_MODEL_UNAVAILABLE = previous_unavailable


if __name__ == "__main__":
    unittest.main()
