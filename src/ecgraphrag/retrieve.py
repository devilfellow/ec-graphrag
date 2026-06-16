from __future__ import annotations

import json
import re
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import Candidate, CommunityReport, Document, Edge, Entity, TextUnit
from .storage import read_jsonl
from .text import (
    bm25_scores,
    content_tokens,
    estimate_tokens,
    lexical_similarity,
    semantic_backend,
    semantic_similarity_scores,
)

_RERANKERS: dict[str, Any] = {}


@dataclass
class RetrievalConfig:
    """Configuration for two-stage document retrieval and reranking."""

    candidate_k: int = 80
    rerank_k: int = 40
    top_k: int = 10
    max_subqueries: int = 4
    rrf_k: int = 60
    use_dense: bool = True
    use_reranker: bool = True
    use_graph: bool = False
    use_enrichment: bool = True
    strict_models: bool = False
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    abstention_threshold: float = 0.30
    weights: dict[str, float] = field(default_factory=lambda: {
        "rrf": 0.45,
        "bm25": 0.20,
        "dense": 0.15,
        "reranker": 0.10,
        "coverage": 0.07,
        "metadata": 0.02,
        "graph": 0.01,
        "enrichment": 0.06,
        "reliability": 0.01,
    })

    @classmethod
    def from_path(cls, path: Path | None) -> "RetrievalConfig":
        """Load retrieval configuration from JSON and merge it with defaults."""
        if not path:
            return cls()
        values = json.loads(path.read_text(encoding="utf-8"))
        config = cls()
        for key, value in values.items():
            if key == "weights" and isinstance(value, dict):
                config.weights.update({str(name): float(weight) for name, weight in value.items()})
            elif hasattr(config, key):
                setattr(config, key, value)
        return config


class Retriever:
    """Load a GraphRAG index and retrieve ranked context for queries."""

    def __init__(
        self,
        index_path: Path,
        weights_path: Path | None = None,
        calibrated: bool = True,
        config: RetrievalConfig | None = None,
    ) -> None:
        """Initialize retrieval state from persisted index tables."""
        self.index_path = index_path
        self.calibrated = calibrated
        self.config = config or RetrievalConfig.from_path(weights_path)
        edges_file = "calibrated_edges.jsonl" if calibrated else "relationships.jsonl"
        self.edges = [Edge(**row) for row in read_jsonl(index_path / edges_file)]
        if not calibrated:
            for edge in self.edges:
                edge.reliability = 1.0
        self.reports = [CommunityReport(**row) for row in read_jsonl(index_path / "community_reports.jsonl")]
        self.units = [TextUnit(**row) for row in read_jsonl(index_path / "text_units.jsonl")]
        self.documents = [Document(**row) for row in read_jsonl(index_path / "documents.jsonl")]
        self.entities = [Entity(**row) for row in read_jsonl(index_path / "entities.jsonl")]
        self.unit_map = {unit.id: unit for unit in self.units}
        self.entity_map = {entity.title.casefold(): entity for entity in self.entities}
        self.edge_map = {edge.id: edge for edge in self.edges}
        self.document_map = {document.id: document for document in self.documents}
        self.document_units: defaultdict[str, list[TextUnit]] = defaultdict(list)
        self.document_edges: defaultdict[str, list[Edge]] = defaultdict(list)
        for unit in self.units:
            self.document_units[unit.document_id].append(unit)
        for edge in self.edges:
            for document_id in edge.source_docs:
                self.document_edges[document_id].append(edge)
        self.hybrid_weights = {
            "bm25": 0.50, "embedding": 0.30, "intent": 0.10,
            "specificity": 0.10,
        }
        if weights_path:
            values = json.loads(weights_path.read_text(encoding="utf-8"))
            if not any(key in values for key in asdict(RetrievalConfig())):
                self.hybrid_weights.update(values)

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        max_hops: int = 2,
        token_budget: int = 1200,
    ) -> dict[str, Any]:
        """Retrieve ranked context for a query using the selected mode."""
        if mode not in {"heuristic", "embedding", "hybrid", "two_stage"}:
            raise ValueError("mode must be heuristic, embedding, hybrid, or two_stage")
        if mode == "two_stage":
            return self._retrieve_two_stage(query, top_k, max_hops, token_budget)
        intent = _intent(query)
        candidates = self._edge_candidates(query, intent, mode)
        candidates.extend(self._path_candidates(query, intent, mode, max_hops))
        candidates.extend(self._report_candidates(query, intent, mode))
        candidates.extend(self._text_candidates(query, mode))
        selected = _select_under_budget(
            candidates,
            top_k,
            token_budget,
            kind_limits={"report": 2, "path": 3},
        )
        return {
            "query": query,
            "mode": mode,
            "calibrated": self.calibrated,
            "intent": intent,
            "token_budget": token_budget,
            "tokens_used": sum(item.token_count for item in selected),
            "context": [asdict(item) for item in selected],
        }

    def _retrieve_two_stage(
        self,
        query: str,
        top_k: int,
        max_hops: int,
        token_budget: int,
    ) -> dict[str, Any]:
        """Rank documents first, then pack document and graph context."""
        started = time.perf_counter()
        ranked_documents, diagnostics = self.rank_documents(query, top_k=top_k)
        ranked_ms = (time.perf_counter() - started) * 1000
        document_context = self._pack_document_context(query, ranked_documents, token_budget)
        remaining = max(0, token_budget - sum(item.token_count for item in document_context))
        graph_context: list[Candidate] = []
        if remaining:
            intent = _intent(query)
            graph_candidates = self._path_candidates(query, intent, "heuristic", max_hops)
            graph_candidates.extend(self._report_candidates(query, intent, "heuristic"))
            graph_context = _select_under_budget(
                graph_candidates,
                top_k=4,
                token_budget=remaining,
                kind_limits={"report": 2, "path": 2},
            )
        context = document_context + graph_context
        return {
            "query": query,
            "mode": "two_stage",
            "calibrated": self.calibrated,
            "intent": _intent(query),
            "token_budget": token_budget,
            "tokens_used": sum(item.token_count for item in context),
            "ranked_documents": ranked_documents,
            "abstained": not ranked_documents or ranked_documents[0]["score"] < self.config.abstention_threshold,
            "context": [asdict(item) for item in context],
            "diagnostics": {
                **diagnostics,
                "rank_documents_ms": round(ranked_ms, 3),
                "total_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        }

    def rank_documents(self, query: str, top_k: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Rank documents with BM25, dense, enrichment, reliability, and reranker features."""
        top_k = top_k or self.config.top_k
        subqueries = decompose_query(query, self.config.max_subqueries)
        titles = [document.title for document in self.documents]
        bodies = [_document_search_text(document, self.document_units[document.id]) for document in self.documents]
        metadata_texts = [_document_metadata_text(document) for document in self.documents]
        graph_texts = (
            [self._document_graph_text(document.id) for document in self.documents]
            if self.config.use_graph else []
        )
        enrichment_fields = self._document_enrichment_fields() if self.config.use_enrichment else []
        scores: defaultdict[str, dict[str, Any]] = defaultdict(lambda: {
            "rrf": 0.0,
            "bm25": 0.0,
            "dense": 0.0,
            "metadata": 0.0,
            "graph": 0.0,
            "enrichment": 0.0,
            "reliability": 0.0,
            "clause_scores": [],
        })
        ranked_lists = 0
        for subquery in subqueries:
            field_scores = [
                ("title", bm25_scores(subquery, titles), 1.25),
                ("body", bm25_scores(subquery, bodies), 1.0),
                ("metadata", bm25_scores(subquery, metadata_texts), 0.80),
            ]
            if self.config.use_graph:
                field_scores.append(("graph", bm25_scores(subquery, graph_texts), 0.20))
            if self.config.use_enrichment:
                for field_name, texts, field_weight in enrichment_fields:
                    if any(texts):
                        field_scores.append((field_name, bm25_scores(subquery, texts), field_weight))
            dense_scores = (
                semantic_similarity_scores(
                    subquery,
                    bodies,
                    model_name=self.config.embedding_model,
                    strict=self.config.strict_models,
                )
                if self.config.use_dense else [0.0] * len(bodies)
            )
            field_scores.append(("dense", dense_scores, 0.75))
            clause_best: defaultdict[str, float] = defaultdict(float)
            for field_name, values, field_weight in field_scores:
                order = sorted(range(len(values)), key=lambda index: (-values[index], self.documents[index].id))
                ranked_lists += 1
                for rank, index in enumerate(order[:self.config.candidate_k], start=1):
                    document_id = self.documents[index].id
                    value = float(values[index])
                    scores[document_id]["rrf"] += field_weight / (self.config.rrf_k + rank)
                    if field_name in {"title", "body"}:
                        scores[document_id]["bm25"] = max(scores[document_id]["bm25"], value)
                    elif field_name == "dense":
                        scores[document_id]["dense"] = max(scores[document_id]["dense"], value)
                    elif field_name == "metadata":
                        scores[document_id]["metadata"] = max(scores[document_id]["metadata"], value)
                    elif field_name == "graph":
                        scores[document_id]["graph"] = max(scores[document_id]["graph"], value)
                    elif field_name in {"questions", "summaries", "entities"}:
                        scores[document_id]["enrichment"] = max(scores[document_id]["enrichment"], value)
                    clause_best[document_id] = max(clause_best[document_id], value)
            for document in self.documents:
                scores[document.id]["clause_scores"].append(clause_best[document.id])

        maximum_rrf = max((row["rrf"] for row in scores.values()), default=1.0)
        candidates: list[dict[str, Any]] = []
        for document in self.documents:
            features = scores[document.id]
            coverage = sum(value >= 0.20 for value in features["clause_scores"]) / max(1, len(subqueries))
            metadata_match = _metadata_match(query, document)
            features["coverage"] = coverage
            features["metadata"] = max(features["metadata"], metadata_match)
            features["reliability"] = self._document_reliability(document.id) if self.calibrated else 0.0
            features["rrf"] /= maximum_rrf
            features["reranker"] = 0.0
            candidates.append({
                "id": document.id,
                "title": document.title,
                "score": 0.0,
                "features": features,
                "best_snippet": _best_snippet(query, self.document_units[document.id], document.text),
            })

        first_stage = sorted(
            candidates,
            key=lambda row: (
                -_weighted_score(row["features"], self.config.weights, exclude={"reranker"}),
                row["id"],
            ),
        )[:self.config.candidate_k]
        rerank_pool = first_stage[:self.config.rerank_k]
        reranker_scores, reranker_backend = _rerank_scores(
            query,
            [f"{row['title']}. {row['best_snippet']}" for row in rerank_pool],
            self.config,
        )
        for row, reranker_score in zip(rerank_pool, reranker_scores):
            row["features"]["reranker"] = reranker_score
        for row in first_stage:
            row["score"] = round(_weighted_score(row["features"], self.config.weights), 6)
        ranked = _coverage_aware_rank(first_stage, subqueries)[:top_k]
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
            row["document_ids"] = [row["id"]]
        return ranked, {
            "subqueries": subqueries,
            "candidate_count": len(first_stage),
            "ranked_lists": ranked_lists,
            "enrichment_fields": [
                name for name, texts, _ in enrichment_fields if any(texts)
            ],
            "embedding_backend": semantic_backend(),
            "reranker_backend": reranker_backend,
        }

    def _pack_document_context(
        self,
        query: str,
        ranked_documents: list[dict[str, Any]],
        token_budget: int,
    ) -> list[Candidate]:
        """Pack ranked documents into context candidates under a token budget."""
        selected: list[Candidate] = []
        remaining = token_budget
        for row in ranked_documents:
            text = f"{row['title']}. {row['best_snippet']}".strip()
            token_count = estimate_tokens(text)
            if token_count > remaining:
                text = _truncate_to_tokens(text, remaining)
                token_count = estimate_tokens(text) if text else 0
            if not text or token_count > remaining:
                continue
            selected.append(Candidate(
                row["id"],
                "document",
                text,
                1.0,
                row["score"],
                row["score"],
                token_count,
                [row["id"]],
                {"document_ids": [row["id"]], "rank": row["rank"], "features": row["features"]},
            ))
            remaining -= token_count
            if remaining <= 0:
                break
        return selected

    def _document_graph_text(self, document_id: str) -> str:
        parts: list[str] = []
        for edge in self.document_edges[document_id]:
            parts.append(edge.edge_text)
            parts.extend(edge.generated_questions[:2])
            if edge.semantic_summary:
                parts.append(edge.semantic_summary)
        return " ".join(parts)

    def _document_enrichment_fields(self) -> list[tuple[str, list[str], float]]:
        """Build separate document-level retrieval fields from enrichment data."""
        questions: list[str] = []
        summaries: list[str] = []
        entities: list[str] = []
        for document in self.documents:
            document_questions: list[str] = []
            document_summaries: list[str] = []
            entity_titles: set[str] = set()
            for edge in self.document_edges[document.id]:
                document_questions.extend(edge.generated_questions)
                if edge.semantic_summary:
                    document_summaries.append(edge.semantic_summary)
                entity_titles.update((edge.source, edge.target))
            entity_parts: list[str] = []
            for title in entity_titles:
                entity = self.entity_map.get(title.casefold())
                if entity:
                    entity_parts.extend([entity.enriched_description, " ".join(entity.aliases), entity.category])
            questions.append(" ".join(document_questions))
            summaries.append(" ".join(document_summaries))
            entities.append(" ".join(part for part in entity_parts if part))
        return [
            ("questions", questions, 0.35),
            ("summaries", summaries, 0.20),
            ("entities", entities, 0.10),
        ]

    def _document_reliability(self, document_id: str) -> float:
        """Average reliability of non-inferred edges attached to a document."""
        edges = [edge for edge in self.document_edges[document_id] if edge.evidence_type != "inferred"]
        if not edges:
            return 0.0
        return sum(edge.reliability for edge in edges) / len(edges)

    def _edge_candidates(self, query: str, intent: str, mode: str) -> list[Candidate]:
        """Build edge candidates for non-two-stage retrieval context."""
        result = []
        texts = [self._edge_retrieval_text(edge) for edge in self.edges]
        semantic = semantic_similarity_scores(query, texts)
        for edge, text, bm25, embedding in zip(self.edges, texts, bm25_scores(query, texts), semantic):
            utility, features = self._utility(query, text, intent, edge.useful_for, mode, bm25, embedding)
            question_boost = _question_boost(query, edge.generated_questions)
            importance_factor = 0.8 + 0.2 * edge.importance
            score = utility * _reliability_factor(edge.reliability) * importance_factor + question_boost
            result.append(Candidate(
                edge.id, "edge", text, edge.reliability, utility,
                score, estimate_tokens(text), [edge.id],
                {**features, "document_ids": edge.source_docs},
            ))
        return result

    def _path_candidates(self, query: str, intent: str, mode: str, max_hops: int) -> list[Candidate]:
        """Build multi-hop path candidates from high-scoring seed edges."""
        if max_hops < 2:
            return []
        adjacency: defaultdict[str, list[Edge]] = defaultdict(list)
        for edge in self.edges:
            adjacency[edge.source].append(edge)
        paths: dict[tuple[str, ...], list[Edge]] = {}
        edge_texts = [self._edge_retrieval_text(edge) for edge in self.edges]
        edge_bm25 = bm25_scores(query, edge_texts)
        seeds = [
            edge for _, edge in sorted(
                zip(edge_bm25, self.edges),
                key=lambda item: item[0],
                reverse=True,
            )[:20]
        ]
        for seed in seeds:
            queue = deque([(seed.target, [seed], {seed.source, seed.target})])
            while queue:
                node, path, visited = queue.popleft()
                if len(path) >= 2:
                    ids = tuple(edge.id for edge in path)
                    paths[ids] = path
                if len(path) >= max_hops:
                    continue
                for edge in adjacency[node]:
                    if edge.id not in {item.id for item in path} and edge.target not in visited:
                        queue.append((edge.target, path + [edge], visited | {edge.target}))
        result = []
        path_rows = [
            (ids, path, " -> ".join(self._edge_retrieval_text(edge) for edge in path))
            for ids, path in paths.items()
        ]
        path_bm25 = bm25_scores(query, [row[2] for row in path_rows])
        path_semantic = semantic_similarity_scores(query, [row[2] for row in path_rows])
        for (ids, path, text), bm25, embedding in zip(path_rows, path_bm25, path_semantic):
            reliability = min(edge.reliability for edge in path)
            useful_for = sorted({tag for edge in path for tag in edge.useful_for})
            utility, features = self._utility(query, text, intent, useful_for, mode, bm25, embedding)
            document_ids = sorted({doc for edge in path for doc in edge.source_docs})
            result.append(Candidate(
                "path:" + ":".join(ids), "path", text, reliability, utility,
                utility * _reliability_factor(reliability) * 1.05,
                estimate_tokens(text), list(ids),
                {**features, "document_ids": document_ids},
            ))
        return result

    def _report_candidates(self, query: str, intent: str, mode: str) -> list[Candidate]:
        """Build community report candidates for global graph context."""
        result = []
        texts = [f"{report.title}. {report.summary}" for report in self.reports]
        semantic = semantic_similarity_scores(query, texts)
        for report, text, bm25, embedding in zip(self.reports, texts, bm25_scores(query, texts), semantic):
            utility, features = self._utility(query, text, intent, ["global"], mode, bm25, embedding)
            document_ids = sorted({
                doc
                for edge_id in report.edge_ids
                for doc in (self.edge_map[edge_id].source_docs if edge_id in self.edge_map else [])
            })
            result.append(Candidate(
                report.id, "report", text, report.reliability, utility,
                utility * _reliability_factor(report.reliability) * 0.65,
                estimate_tokens(text), report.edge_ids,
                {**features, "document_ids": document_ids},
            ))
        return result

    def _text_candidates(self, query: str, mode: str) -> list[Candidate]:
        """Build text-unit candidates for legacy retrieval modes."""
        result = []
        texts = [unit.text for unit in self.units]
        semantic = semantic_similarity_scores(query, texts)
        for unit, bm25, embedding in zip(self.units, bm25_scores(query, texts), semantic):
            lexical = lexical_similarity(query, unit.text)
            utility = embedding if mode == "embedding" else bm25
            if mode == "hybrid":
                utility = 0.65 * bm25 + 0.35 * embedding
            result.append(Candidate(
                unit.id, "text_unit", unit.text, 0.7, utility, utility * 0.90,
                unit.token_count, [unit.id],
                {"bm25": bm25, "lexical": lexical, "embedding": embedding, "document_ids": [unit.document_id]},
            ))
        return result

    def _utility(
        self,
        query: str,
        text: str,
        intent: str,
        useful_for: list[str],
        mode: str,
        bm25: float,
        embedding: float,
    ) -> tuple[float, dict[str, float]]:
        """Combine lexical, semantic, intent, and specificity features."""
        lexical = lexical_similarity(query, text)
        intent_match = 1.0 if intent in useful_for else 0.35
        specificity = min(1.0, 12 / max(12, estimate_tokens(text)))
        features = {
            "bm25": bm25,
            "lexical": lexical,
            "embedding": embedding,
            "intent": intent_match,
            "specificity": specificity,
        }
        if mode == "embedding":
            utility = 0.8 * embedding + 0.2 * intent_match
        elif mode == "hybrid":
            utility = sum(self.hybrid_weights.get(key, 0.0) * value for key, value in features.items())
        else:
            utility = 0.65 * bm25 + 0.20 * intent_match + 0.15 * specificity
        return round(max(0.0, min(1.0, utility)), 6), features

    def _edge_retrieval_text(self, edge: Edge) -> str:
        """Build retrieval text for an edge and its connected enriched entities."""
        entity_parts: list[str] = []
        for title in (edge.source, edge.target):
            entity = self.entity_map.get(title.casefold())
            if entity:
                entity_parts.extend([entity.enriched_description, " ".join(entity.aliases), entity.category])
        return " ".join(
            part for part in (
                edge.edge_text,
                edge.semantic_summary,
                edge.evidence_text,
                " ".join(entity_parts),
            )
            if part
        )


def _intent(query: str) -> str:
    lowered = query.casefold()
    if any(word in lowered for word in ("why", "cause", "reason", "почему", "причин")):
        return "causal"
    if any(word in lowered for word in ("where", "location", "где", "мест")):
        return "factual"
    if any(word in lowered for word in ("overview", "theme", "summar", "обзор", "тем", "итог")):
        return "global"
    return "exploratory"


def _question_boost(query: str, questions: list[str]) -> float:
    """Boost score when query closely matches a generated question."""
    if not questions:
        return 0.0
    best = max(lexical_similarity(query, q) for q in questions)
    return 0.15 * best


def _edge_retrieval_text(edge: Edge) -> str:
    return " ".join(part for part in (edge.edge_text, edge.semantic_summary) if part)


def _reliability_factor(reliability: float) -> float:
    return 0.75 + 0.25 * max(0.0, min(1.0, reliability))


def _select_under_budget(
    candidates: list[Candidate],
    top_k: int,
    token_budget: int,
    kind_limits: dict[str, int] | None = None,
) -> list[Candidate]:
    """Select high-scoring non-duplicate candidates under token and kind limits."""
    selected: list[Candidate] = []
    used_ids: set[str] = set()
    remaining = token_budget
    kind_counts: defaultdict[str, int] = defaultdict(int)
    kind_limits = kind_limits or {}
    ranked = sorted(candidates, key=lambda item: (-item.score, item.token_count, item.id))
    while ranked and len(selected) < top_k:
        best: Candidate | None = None
        best_adjusted = -1.0
        for candidate in ranked:
            if candidate.token_count > remaining:
                continue
            if kind_counts[candidate.kind] >= kind_limits.get(candidate.kind, top_k):
                continue
            overlap = len(set(candidate.source_ids) & used_ids) / max(1, len(candidate.source_ids))
            adjusted = candidate.score * (1.0 - 0.65 * overlap)
            if adjusted > best_adjusted:
                best, best_adjusted = candidate, adjusted
        if best is None:
            break
        selected.append(best)
        used_ids.update(best.source_ids)
        kind_counts[best.kind] += 1
        remaining -= best.token_count
        ranked.remove(best)
    return selected


def decompose_query(query: str, max_subqueries: int = 4) -> list[str]:
    """Split a query into the full query plus useful multi-hop clauses."""
    normalized = re.sub(r"\s+", " ", query).strip()
    if not normalized:
        return []
    parts = re.split(
        r"\s*(?:,?\s+(?:and|while|whereas|as well as|according to|as reported by)\s+|[;])\s*",
        normalized,
        flags=re.IGNORECASE,
    )
    result = [normalized]
    for part in parts:
        part = part.strip(" ,.?")
        if len(content_tokens(part)) >= 4 and part.casefold() != normalized.casefold():
            result.append(part)
        if len(result) >= max_subqueries:
            break
    return result


def _document_search_text(document: Document, units: list[TextUnit]) -> str:
    unit_text = " ".join(unit.text for unit in units)
    return f"{document.title}. {unit_text or document.text}"


def _document_metadata_text(document: Document) -> str:
    metadata = document.metadata
    nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
    values = [
        document.title,
        str(metadata.get("source") or nested.get("source") or ""),
        str(metadata.get("category") or nested.get("category") or ""),
        str(metadata.get("published_at") or nested.get("published_at") or ""),
        str(metadata.get("author") or nested.get("author") or ""),
    ]
    return " ".join(value for value in values if value)


def _metadata_match(query: str, document: Document) -> float:
    query_tokens = set(content_tokens(query))
    metadata_tokens = set(content_tokens(_document_metadata_text(document)))
    if not query_tokens or not metadata_tokens:
        return 0.0
    return len(query_tokens & metadata_tokens) / len(metadata_tokens)


def _best_snippet(query: str, units: list[TextUnit], document_text: str) -> str:
    if units:
        scores = bm25_scores(query, [unit.text for unit in units])
        best_index = max(range(len(units)), key=lambda index: scores[index])
        return _truncate_to_tokens(units[best_index].text, 180)
    return _truncate_to_tokens(document_text, 180)


def _truncate_to_tokens(text: str, token_limit: int) -> str:
    if token_limit <= 0:
        return ""
    words = text.split()
    approximate_words = max(1, int(token_limit / 1.25))
    return " ".join(words[:approximate_words])


def _weighted_score(
    features: dict[str, Any],
    weights: dict[str, float],
    exclude: set[str] | None = None,
) -> float:
    """Compute a weighted feature score with optional excluded features."""
    exclude = exclude or set()
    return sum(
        weights.get(name, 0.0) * float(features.get(name, 0.0))
        for name in weights
        if name not in exclude
    )


def _rerank_scores(
    query: str,
    documents: list[str],
    config: RetrievalConfig,
) -> tuple[list[float], str]:
    """Score query-document pairs with a CrossEncoder or lexical fallback."""
    if not documents:
        return [], "none"
    if config.use_reranker:
        try:
            from sentence_transformers import CrossEncoder

            model = _RERANKERS.get(config.reranker_model)
            if model is None:
                model = CrossEncoder(config.reranker_model)
                _RERANKERS[config.reranker_model] = model
            values = model.predict([(query, document) for document in documents])
            raw = [float(value) for value in values]
            minimum = min(raw)
            maximum = max(raw)
            normalized = [
                (value - minimum) / (maximum - minimum) if maximum > minimum else 0.5
                for value in raw
            ]
            return normalized, config.reranker_model
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            if config.strict_models:
                raise RuntimeError(f"Reranker model is unavailable: {config.reranker_model}") from exc
    return [lexical_similarity(query, document) for document in documents], "lexical-fallback"


def _coverage_aware_rank(
    candidates: list[dict[str, Any]],
    subqueries: list[str],
) -> list[dict[str, Any]]:
    """Prefer high-scoring documents that add coverage for query clauses."""
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    covered = [0.0] * len(subqueries)
    if remaining:
        first = max(remaining, key=lambda row: (row["score"], row["id"]))
        selected.append(first)
        for index, value in enumerate(first["features"]["clause_scores"]):
            covered[index] = max(covered[index], float(value))
        remaining.remove(first)
    while remaining:
        best = max(
            remaining,
            key=lambda row: (
                row["score"] + 0.08 * sum(
                    max(0.0, float(value) - covered[index])
                    for index, value in enumerate(row["features"]["clause_scores"])
                ),
                row["id"],
            ),
        )
        selected.append(best)
        for index, value in enumerate(best["features"]["clause_scores"]):
            covered[index] = max(covered[index], float(value))
        remaining.remove(best)
    return selected
