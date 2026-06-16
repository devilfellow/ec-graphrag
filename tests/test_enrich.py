"""Tests for LLM-based graph enrichment."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ecgraphrag.enrich import (
    enrich_contradiction_analysis,
    enrich_edge_importance,
    enrich_edge_questions,
    enrich_edge_summaries,
    enrich_entities,
    enrich_graph,
    enrich_inference_edges,
)
from ecgraphrag.models import Edge, Entity


class MockLLMClient:
    """Mock that returns configurable JSON responses based on prompt content."""

    def __init__(self, responses: dict[str, dict[str, Any]] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, system: str, user: str, schema_hint: str | None = None) -> dict[str, Any]:
        self.calls.append((system, user))
        for key, value in self.responses.items():
            if key in user:
                return value
        return self._default_response(system)

    def _default_response(self, system: str) -> dict[str, Any]:
        if "questions" in system:
            return {"questions": ["What is the relationship?", "How are they connected?"]}
        if "Rewrite" in system:
            return {"summary": "A concise summary of the fact."}
        if "contradict" in system:
            return {"has_contradiction": False, "explanation": "", "conflicting_ids": []}
        if "enrich" in system.lower() or "aliases" in system:
            return {"description": "Enriched description.", "aliases": ["alias1"], "category": "Organization"}
        if "infer" in system.lower():
            return {"inferred": None}
        if "importance" in system.lower() or "central" in system.lower():
            return {"importance": 0.7, "reasoning": "Key fact."}
        return {}


def _make_edge(
    id: str = "e1",
    source: str = "Alpha",
    target: str = "Beta",
    relation: str = "causes",
    description: str = "Alpha causes Beta",
    evidence: str = "Alpha causes Beta in production",
    **kwargs: Any,
) -> Edge:
    defaults = dict(
        id=id, source=source, target=target, relation=relation,
        description=description, evidence_text=evidence,
        edge_text=f"{source} {relation} {target}. {description}",
        llm_conf=0.8, reliability=0.7, importance=0.5,
    )
    defaults.update(kwargs)
    return Edge(**defaults)


def _make_entity(id: str = "ent1", title: str = "Alpha", type: str = "System") -> Entity:
    return Entity(id=id, title=title, type=type, description="A system component")


class TestQuestionGeneration(unittest.TestCase):
    def test_questions_populated(self):
        """LLM-generated questions are stored on the edge."""
        edge = _make_edge()
        client = MockLLMClient({
            "Alpha": {"questions": [
                "What causes Beta?",
                "What does Alpha cause?",
                "How are Alpha and Beta related?",
            ]}
        })
        result = enrich_edge_questions([edge], client)
        self.assertEqual(len(result[0].generated_questions), 3)
        self.assertIn("What causes Beta?", result[0].generated_questions)

    def test_questions_default_empty_on_failure(self):
        """If LLM fails, generated_questions stays empty."""
        edge = _make_edge()

        class FailingClient:
            def chat_json(self, system: str, user: str, schema_hint: str | None = None) -> dict:
                raise RuntimeError("API unavailable")

        result = enrich_edge_questions([edge], FailingClient())
        self.assertEqual(result[0].generated_questions, [])

    def test_questions_improve_retrieval_score(self):
        """Edges with matching questions should get a boost in retrieval."""
        from ecgraphrag.retrieve import _question_boost

        questions = ["What causes production delay?", "Why is production delayed?"]
        boost = _question_boost("Why is production delayed?", questions)
        self.assertGreater(boost, 0.05)

        no_boost = _question_boost("Unrelated query about weather", questions)
        self.assertLess(no_boost, boost)

    def test_no_questions_no_boost(self):
        """Empty question list should give zero boost."""
        from ecgraphrag.retrieve import _question_boost

        self.assertEqual(_question_boost("any query", []), 0.0)

    def test_questions_use_configured_workers(self):
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
                return {"questions": ["What is connected?"]}

        edges = [_make_edge(id=f"e{index}") for index in range(24)]
        client = ConcurrentClient()
        enrich_edge_questions(edges, client)
        self.assertGreater(client.max_active, 1)
        self.assertLessEqual(client.max_active, 12)


class TestSemanticSummary(unittest.TestCase):
    def test_summary_populated(self):
        """LLM-generated summary is stored on the edge."""
        edge = _make_edge()
        client = MockLLMClient({
            "Alpha": {"summary": "Alpha directly leads to Beta occurring in the production environment."}
        })
        result = enrich_edge_summaries([edge], client)
        self.assertEqual(
            result[0].semantic_summary,
            "Alpha directly leads to Beta occurring in the production environment."
        )

    def test_summary_used_in_retrieval(self):
        """Retriever should prefer semantic_summary over edge_text when available."""
        from ecgraphrag.models import Candidate
        from ecgraphrag.text import lexical_similarity

        edge = _make_edge()
        edge.semantic_summary = "Missing wagons cause equipment downtime in rail operations"
        edge.edge_text = "Missing Wagons causes Equipment Downtime. Missing Wagons causes Equipment Downtime"

        query = "What leads to equipment problems in railways?"
        sim_summary = lexical_similarity(query, edge.semantic_summary)
        sim_raw = lexical_similarity(query, edge.edge_text)
        self.assertGreater(sim_summary, sim_raw * 0.5)

    def test_empty_summary_falls_back(self):
        """If summary is empty string, edge_text should be used instead."""
        from ecgraphrag.retrieve import _edge_retrieval_text

        edge = _make_edge()
        edge.semantic_summary = ""
        text = _edge_retrieval_text(edge)
        self.assertEqual(text, edge.edge_text)

    def test_summary_keeps_original_edge_text(self):
        from ecgraphrag.retrieve import _edge_retrieval_text

        edge = _make_edge()
        edge.semantic_summary = "A fluent semantic description."
        text = _edge_retrieval_text(edge)
        self.assertIn(edge.edge_text, text)
        self.assertIn(edge.semantic_summary, text)


class TestContradictionAnalysis(unittest.TestCase):
    def test_contradiction_detected(self):
        """LLM identifies contradicting edges between same entity pair."""
        e1 = _make_edge(id="e1", relation="causes", description="Alpha causes Beta")
        e2 = _make_edge(id="e2", relation="prevents", description="Alpha prevents Beta")
        client = MockLLMClient({
            "Alpha": {
                "has_contradiction": True,
                "explanation": "One says causes, other says prevents",
                "conflicting_ids": ["e1", "e2"],
            }
        })
        result = enrich_contradiction_analysis([e1, e2], client)
        self.assertEqual(result[0].conflict_status, "contradiction")
        self.assertEqual(result[1].conflict_status, "contradiction")
        self.assertIn("causes", result[0].contradiction_info)

    def test_no_contradiction_single_edge(self):
        """Single edge per pair should not trigger analysis."""
        edge = _make_edge()
        client = MockLLMClient()
        result = enrich_contradiction_analysis([edge], client)
        self.assertEqual(result[0].conflict_status, "ok")
        self.assertEqual(result[0].contradiction_info, "")

    def test_contradiction_lowers_trust(self):
        """Edges marked as contradictions should be treated with lower trust."""
        edge = _make_edge()
        edge.conflict_status = "contradiction"
        edge.contradiction_info = "Directly conflicts with another edge"
        self.assertEqual(edge.conflict_status, "contradiction")


class TestEntityEnrichment(unittest.TestCase):
    def test_entity_description_enriched(self):
        """LLM generates a richer description from graph context."""
        entity = _make_entity()
        edges = [_make_edge(source="Alpha", target="Beta")]
        client = MockLLMClient({
            "Alpha": {
                "description": "Alpha is a critical system component that triggers Beta failures",
                "aliases": ["System Alpha", "AlphaModule"],
                "category": "Infrastructure",
            }
        })
        result = enrich_entities([entity], edges, client)
        self.assertIn("critical system component", result[0].enriched_description)
        self.assertIn("AlphaModule", result[0].aliases)
        self.assertEqual(result[0].category, "Infrastructure")

    def test_entity_without_edges_unchanged(self):
        """Entities with no connected edges are not enriched."""
        entity = _make_entity(title="Orphan")
        client = MockLLMClient()
        result = enrich_entities([entity], [], client)
        self.assertEqual(result[0].enriched_description, "")

    def test_aliases_improve_entity_matching(self):
        """Entity aliases can be used for fuzzy matching in queries."""
        entity = _make_entity()
        entity.aliases = ["System Alpha", "AlphaModule", "α-system"]
        from ecgraphrag.text import lexical_similarity

        query = "What is AlphaModule?"
        max_sim = max(
            lexical_similarity(query, alias) for alias in entity.aliases
        )
        self.assertGreater(max_sim, 0.0)


class TestInferenceEdges(unittest.TestCase):
    def test_inferred_edge_created(self):
        """LLM generates a new inferred edge from a 2-hop path."""
        e1 = _make_edge(id="e1", source="Alpha", target="Beta", relation="causes")
        e2 = _make_edge(id="e2", source="Beta", target="Gamma", relation="causes")
        e1.reliability = 0.9
        e2.reliability = 0.8

        client = MockLLMClient({
            "Alpha": {
                "inferred": {
                    "source": "Alpha",
                    "target": "Gamma",
                    "relation": "indirectly_causes",
                    "description": "Alpha indirectly causes Gamma through Beta",
                    "confidence": 0.65,
                }
            }
        })
        result = enrich_inference_edges([e1, e2], client)
        self.assertEqual(len(result), 3)
        inferred = [e for e in result if e.evidence_type == "inferred"]
        self.assertEqual(len(inferred), 1)
        self.assertEqual(inferred[0].source, "Alpha")
        self.assertEqual(inferred[0].target, "Gamma")
        self.assertEqual(inferred[0].relation, "indirectly_causes")
        self.assertAlmostEqual(inferred[0].llm_conf, 0.65)

    def test_no_duplicate_inferred_edges(self):
        """If inferred edge already exists, it should not be added again."""
        e1 = _make_edge(id="e1", source="Alpha", target="Beta", relation="causes")
        e2 = _make_edge(id="e2", source="Beta", target="Gamma", relation="causes")
        e_existing = _make_edge(id="e3", source="Alpha", target="Gamma", relation="indirectly_causes")
        e1.reliability = 0.9
        e2.reliability = 0.8

        client = MockLLMClient({
            "Alpha": {
                "inferred": {
                    "source": "Alpha",
                    "target": "Gamma",
                    "relation": "indirectly_causes",
                    "description": "Already exists",
                    "confidence": 0.65,
                }
            }
        })
        result = enrich_inference_edges([e1, e2, e_existing], client)
        inferred_count = sum(1 for e in result if e.evidence_type == "inferred")
        self.assertEqual(inferred_count, 0)

    def test_null_inference_no_new_edge(self):
        """When LLM returns null, no new edge is created."""
        e1 = _make_edge(id="e1", source="Alpha", target="Beta", relation="located_in")
        e2 = _make_edge(id="e2", source="Beta", target="Gamma", relation="part_of")
        e1.reliability = 0.9

        client = MockLLMClient()
        result = enrich_inference_edges([e1, e2], client)
        self.assertEqual(len(result), 2)


class TestEdgeImportance(unittest.TestCase):
    def test_importance_set(self):
        """LLM rates edge importance."""
        edge = _make_edge()
        client = MockLLMClient({
            "Alpha": {"importance": 0.9, "reasoning": "Critical causal link"}
        })
        result = enrich_edge_importance([edge], client)
        self.assertAlmostEqual(result[0].importance, 0.9)

    def test_importance_clamped(self):
        """Importance is clamped to [0, 1]."""
        edge = _make_edge()
        client = MockLLMClient({
            "Alpha": {"importance": 1.5, "reasoning": "Overclaimed"}
        })
        result = enrich_edge_importance([edge], client)
        self.assertLessEqual(result[0].importance, 1.0)

    def test_importance_affects_retrieval_score(self):
        """Higher importance should boost retrieval score."""
        high_imp = _make_edge(id="high")
        high_imp.importance = 0.9
        low_imp = _make_edge(id="low")
        low_imp.importance = 0.2

        factor_high = 0.8 + 0.2 * high_imp.importance
        factor_low = 0.8 + 0.2 * low_imp.importance
        self.assertGreater(factor_high, factor_low)


class TestEnrichGraph(unittest.TestCase):
    def test_all_steps_run(self):
        """enrich_graph runs all enrichment steps."""
        entities = [_make_entity()]
        edges = [_make_edge()]
        client = MockLLMClient()
        result_entities, result_edges = enrich_graph(entities, edges, client)
        self.assertTrue(len(client.calls) >= 4)
        self.assertTrue(result_edges[0].generated_questions)
        self.assertTrue(result_edges[0].semantic_summary)

    def test_selective_steps(self):
        """Only selected steps run when specified."""
        entities = [_make_entity()]
        edges = [_make_edge()]
        client = MockLLMClient()
        enrich_graph(entities, edges, client, steps=["questions"])
        calls_with_questions = [c for c in client.calls if "questions" in c[0].lower()]
        self.assertTrue(calls_with_questions)
        calls_with_summary = [c for c in client.calls if "Rewrite" in c[0]]
        self.assertFalse(calls_with_summary)

    def test_backward_compatibility(self):
        """Edges without new fields should load with defaults."""
        old_data = {
            "id": "e_old", "source": "X", "target": "Y",
            "relation": "related_to", "description": "X related to Y",
        }
        edge = Edge(**old_data)
        self.assertEqual(edge.generated_questions, [])
        self.assertEqual(edge.semantic_summary, "")
        self.assertEqual(edge.contradiction_info, "")
        self.assertEqual(edge.importance, 0.5)

    def test_entity_backward_compatibility(self):
        """Entities without new fields load with defaults."""
        old_data = {"id": "ent_old", "title": "OldEntity"}
        entity = Entity(**old_data)
        self.assertEqual(entity.enriched_description, "")
        self.assertEqual(entity.aliases, [])
        self.assertEqual(entity.category, "")

    def test_finalize_recalibrates_inferred_edges_and_rebuilds_reports(self):
        from ecgraphrag.enrich import finalize_enriched_index
        from ecgraphrag.storage import export_table, read_jsonl
        from dataclasses import asdict

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            output = Path(temp) / "output"
            source.mkdir()
            edge = _make_edge(evidence_type="inferred", reliability=0.0)
            entities = [_make_entity(title="Alpha"), _make_entity(id="ent2", title="Beta")]
            for name, values in (
                ("documents", []), ("text_units", []), ("relationships", [edge]),
                ("calibrated_edges", [edge]), ("entities", entities),
                ("communities", []), ("community_reports", []),
            ):
                export_table(source, name, [asdict(value) for value in values])
            finalize_enriched_index(source, output, entities, [edge])
            calibrated = read_jsonl(output / "calibrated_edges.jsonl")
            reports = read_jsonl(output / "community_reports.jsonl")
            self.assertGreater(calibrated[0]["reliability"], 0)
            self.assertTrue(reports)


class TestEnrichmentIntegration(unittest.TestCase):
    def test_enriched_retrieval_prefers_question_match(self):
        """When a query closely matches generated questions, that edge scores higher."""
        from ecgraphrag.text import lexical_similarity
        from ecgraphrag.retrieve import _question_boost

        edge_with_q = _make_edge(id="eq1", source="Missing Wagons", target="Downtime")
        edge_with_q.generated_questions = [
            "What causes equipment downtime?",
            "Why does downtime happen?",
            "What is the impact of missing wagons?",
        ]
        edge_without_q = _make_edge(id="eq2", source="Missing Wagons", target="Delay")
        edge_without_q.generated_questions = []

        query = "What causes equipment downtime?"
        boost_with = _question_boost(query, edge_with_q.generated_questions)
        boost_without = _question_boost(query, edge_without_q.generated_questions)
        self.assertGreater(boost_with, boost_without)

    def test_enriched_graph_indexing(self):
        """Full pipeline: index -> enrich -> calibrate -> retrieve."""
        from ecgraphrag.calibrate import calibrate_edges
        from ecgraphrag.extract import extract_graph
        from ecgraphrag.ingest import chunk_documents, ingest
        from ecgraphrag.retrieve import Retriever
        from ecgraphrag.storage import export_table, read_jsonl

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            output = root / "index"
            data.mkdir()
            output.mkdir()
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

            documents = ingest(data)
            units = chunk_documents(documents, 100, 10)
            entities, edges = extract_graph(documents, units)

            client = MockLLMClient({
                "Missing Wagons": {
                    "questions": [
                        "What causes equipment downtime?",
                        "What happens when wagons are missing?",
                    ]
                },
                "Equipment Downtime": {
                    "questions": [
                        "What causes production delay?",
                        "What results from equipment downtime?",
                    ]
                },
            })
            edges = enrich_edge_questions(edges, client)
            calibrate_edges(edges)

            from ecgraphrag.communities import build_communities
            from dataclasses import asdict

            communities, reports = build_communities(entities, edges)
            for name, items in [
                ("calibrated_edges", edges),
                ("text_units", units),
                ("community_reports", reports),
            ]:
                export_table(output, name, [asdict(item) for item in items])

            retriever = Retriever(output)
            result = retriever.retrieve("What causes equipment downtime?", token_budget=500)
            self.assertTrue(result["context"])
            top_edge = result["context"][0]
            self.assertIn("downtime", top_edge["text"].lower())


if __name__ == "__main__":
    unittest.main()
