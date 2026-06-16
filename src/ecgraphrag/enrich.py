"""LLM-based enrichment for graph nodes and edges."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Protocol

from .calibrate import calibrate_edges
from .communities import build_communities
from .extract import _normalize_relation
from .models import Edge, Entity
from .storage import export_table, write_json
from .text import stable_id

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for clients that return JSON objects from chat prompts."""

    def chat_json(self, system: str, user: str, schema_hint: str | None = None) -> dict[str, Any]:
        """Return a JSON object for the supplied chat prompt."""
        ...


_QG_SYSTEM = (
    "You generate natural-language questions that the given knowledge graph fact can answer. "
    "Use only the supplied fact and evidence; do not add outside knowledge. "
    "Each question must be answerable from this single fact and preserve entity names. "
    "Return JSON: {\"questions\": [\"...\", ...]}. Generate 3-5 diverse questions."
)

_QG_SCHEMA = '{"questions": ["string"]}'


def enrich_edge_questions(edges: list[Edge], client: LLMClient, batch_size: int = 10) -> list[Edge]:
    """For each edge, ask LLM to generate questions this edge answers."""
    def process(edge: Edge) -> None:
        prompt = (
            f"Fact: {edge.source} {edge.relation.replace('_', ' ')} {edge.target}.\n"
            f"Description: {edge.description}\n"
            f"Evidence: {edge.evidence_text}"
        )
        try:
            result = client.chat_json(_QG_SYSTEM, prompt, _QG_SCHEMA)
            questions = result.get("questions", [])
            if isinstance(questions, list):
                edge.generated_questions = [str(q) for q in questions if q]
        except Exception as exc:
            logger.warning("Question enrichment failed for edge %s: %s", edge.id, exc)
    _run_parallel(edges, process, client)
    return edges


_SS_SYSTEM = (
    "Rewrite the following knowledge graph triple and its evidence into a concise, "
    "natural-language sentence optimized for information retrieval. Preserve both entity names "
    "and every claim from the triple. Use only the supplied evidence and do not add outside knowledge. "
    "Return JSON: {\"summary\": \"...\"}."
)

_SS_SCHEMA = '{"summary": "string"}'


def enrich_edge_summaries(edges: list[Edge], client: LLMClient) -> list[Edge]:
    """Replace mechanical edge_text with LLM-generated fluent summary."""
    def process(edge: Edge) -> None:
        prompt = (
            f"Triple: ({edge.source}, {edge.relation}, {edge.target})\n"
            f"Evidence: {edge.evidence_text}\n"
            f"Description: {edge.description}"
        )
        try:
            result = client.chat_json(_SS_SYSTEM, prompt, _SS_SCHEMA)
            summary = result.get("summary", "")
            if summary:
                edge.semantic_summary = str(summary)
        except Exception as exc:
            logger.warning("Summary enrichment failed for edge %s: %s", edge.id, exc)
    _run_parallel(edges, process, client)
    return edges


_CA_SYSTEM = (
    "You are given multiple knowledge graph edges between the same pair of entities. "
    "Determine if any of them directly contradict each other. Different relations are not "
    "automatically contradictions. Use only the supplied facts. "
    "Return JSON: {\"has_contradiction\": bool, \"explanation\": \"...\", "
    "\"conflicting_ids\": [\"edge_id_1\", \"edge_id_2\"]}."
)

_CA_SCHEMA = '{"has_contradiction": false, "explanation": "string", "conflicting_ids": []}'


def enrich_contradiction_analysis(edges: list[Edge], client: LLMClient) -> list[Edge]:
    """For each entity pair with multiple edges, ask LLM to detect contradictions."""
    pairs: defaultdict[tuple[str, str], list[Edge]] = defaultdict(list)
    for edge in edges:
        key = (edge.source.casefold(), edge.target.casefold())
        pairs[key].append(edge)

    groups = [(pair_key, group) for pair_key, group in pairs.items() if len(group) >= 2]

    def process(item: tuple[tuple[str, str], list[Edge]]) -> None:
        pair_key, group = item
        if len(group) < 2:
            return
        facts = "\n".join(
            f"- [{e.id}] {e.source} {e.relation} {e.target}: {e.description}"
            for e in group
        )
        prompt = f"Entity pair: ({group[0].source}, {group[0].target})\nEdges:\n{facts}"
        try:
            result = client.chat_json(_CA_SYSTEM, prompt, _CA_SCHEMA)
            if result.get("has_contradiction"):
                explanation = str(result.get("explanation", ""))
                conflicting = set(result.get("conflicting_ids", []))
                for edge in group:
                    if edge.id in conflicting:
                        edge.contradiction_info = explanation
                        edge.conflict_status = "contradiction"
        except Exception as exc:
            logger.warning("Contradiction enrichment failed for pair %s: %s", pair_key, exc)
    _run_parallel(groups, process, client)
    return edges


_EE_SYSTEM = (
    "You enrich a knowledge graph entity with a clear, comprehensive description "
    "based only on its supplied graph context. Do not use outside knowledge. "
    "Aliases must appear in the supplied context or be an unambiguous abbreviation of the entity name. "
    "Also provide a high-level category. "
    "Return JSON: {\"description\": \"...\", \"aliases\": [\"...\"], \"category\": \"...\"}."
)

_EE_SCHEMA = '{"description": "string", "aliases": ["string"], "category": "string"}'


def enrich_entities(
    entities: list[Entity], edges: list[Edge], client: LLMClient
) -> list[Entity]:
    """For each entity, generate enriched description from its graph context."""
    entity_edges: defaultdict[str, list[str]] = defaultdict(list)
    for edge in edges:
        entity_edges[edge.source.casefold()].append(
            f"{edge.source} {edge.relation} {edge.target}: {edge.description}"
        )
        entity_edges[edge.target.casefold()].append(
            f"{edge.source} {edge.relation} {edge.target}: {edge.description}"
        )

    def process(entity: Entity) -> None:
        context_lines = entity_edges.get(entity.title.casefold(), [])
        if not context_lines:
            return
        context = "\n".join(context_lines[:10])
        prompt = (
            f"Entity: {entity.title}\n"
            f"Type: {entity.type}\n"
            f"Current description: {entity.description}\n"
            f"Graph context (relationships):\n{context}"
        )
        try:
            result = client.chat_json(_EE_SYSTEM, prompt, _EE_SCHEMA)
            desc = result.get("description", "")
            if desc:
                entity.enriched_description = str(desc)
            aliases = result.get("aliases", [])
            if isinstance(aliases, list):
                entity.aliases = [str(a) for a in aliases if a]
            category = result.get("category", "")
            if category:
                entity.category = str(category)
        except Exception as exc:
            logger.warning("Entity enrichment failed for %s: %s", entity.id, exc)
    _run_parallel(entities, process, client)
    return entities


_IE_SYSTEM = (
    "You are given a chain of knowledge graph facts (a path). "
    "Infer a NEW direct relationship only when it follows necessarily from both facts. "
    "Do not use outside knowledge. The source and target must be the requested path endpoints. "
    "If the inference is uncertain, merely associated, or not transitive, return null. "
    "Return JSON: {\"inferred\": {\"source\": \"...\", \"target\": \"...\", "
    "\"relation\": \"...\", \"description\": \"...\", \"confidence\": 0.0} | null}."
)

_IE_SCHEMA = (
    '{"inferred": {"source": "string", "target": "string", '
    '"relation": "string", "description": "string", "confidence": 0.0}}'
)


def enrich_inference_edges(
    edges: list[Edge],
    client: LLMClient,
    max_hops: int = 2,
    max_paths: int | None = 200,
) -> list[Edge]:
    """Generate new inferred edges from multi-hop paths."""
    adjacency: defaultdict[str, list[Edge]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.source.casefold()].append(edge)

    existing_keys = {
        (e.source.casefold(), e.target.casefold(), e.relation) for e in edges
    }
    inferred: list[Edge] = []
    seen_paths: set[tuple[str, ...]] = set()

    path_tasks: list[tuple[Edge, Edge, tuple[str, ...]]] = []
    seeds = sorted(edges, key=lambda e: e.reliability, reverse=True)
    for seed in seeds:
        node = seed.target.casefold()
        for next_edge in adjacency.get(node, [])[:5]:
            if max_paths is not None and len(seen_paths) >= max_paths:
                break
            path_key = (seed.id, next_edge.id)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            path_tasks.append((seed, next_edge, path_key))
        if max_paths is not None and len(seen_paths) >= max_paths:
            break

    def infer_path(task: tuple[Edge, Edge, tuple[str, ...]]) -> tuple[Edge, Edge, dict[str, Any] | None]:
        seed, next_edge, path_key = task
        try:
            chain_text = (
                f"1. {seed.source} {seed.relation} {seed.target}: {seed.description}\n"
                f"2. {next_edge.source} {next_edge.relation} {next_edge.target}: {next_edge.description}"
            )
            prompt = f"Path:\n{chain_text}\n\nCan you infer a direct relationship between {seed.source} and {next_edge.target}?"
            result = client.chat_json(_IE_SYSTEM, prompt, _IE_SCHEMA)
            inf = result.get("inferred")
            return seed, next_edge, inf if isinstance(inf, dict) else None
        except Exception as exc:
            logger.warning("Inference enrichment failed for path %s: %s", path_key, exc)
            return seed, next_edge, None

    for seed, next_edge, inf in _run_parallel(path_tasks, infer_path, client):
        if inf:
            source = seed.source
            target = next_edge.target
            relation, ontology_ok = _normalize_relation(str(inf.get("relation", "associated_with")))
            key = (source.casefold(), target.casefold(), relation)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            description = str(inf.get("description", ""))
            confidence = float(inf.get("confidence", 0.5))
            edge_id = stable_id("inferred", *key)
            new_edge = Edge(
                id=edge_id,
                source=source,
                target=target,
                relation=relation,
                description=description,
                evidence_text=f"Inferred from: {seed.edge_text} + {next_edge.edge_text}",
                evidence_type="inferred",
                source_docs=list(set(seed.source_docs + next_edge.source_docs)),
                text_unit_ids=list(set(seed.text_unit_ids + next_edge.text_unit_ids)),
                llm_conf=max(0.0, min(1.0, confidence)),
                ontology_ok=ontology_ok,
                reliability=min(seed.reliability, next_edge.reliability) * max(0.0, min(1.0, confidence)),
                edge_text=f"{source} {relation.replace('_', ' ')} {target}. {description}",
                edge_emb_id=f"emb_{edge_id}",
                useful_for=["causal", "exploratory"],
                importance=confidence * 0.8,
            )
            inferred.append(new_edge)

    return edges + inferred


_EI_SYSTEM = (
    "Rate how important/central this knowledge graph fact is for understanding the topic. "
    "Use only the supplied fact and evidence. Consider whether it is a key fact or a trivial detail. "
    "Return JSON: {\"importance\": 0.0, \"reasoning\": \"...\"}. "
    "Importance should be 0.0-1.0 (1.0 = absolutely critical fact)."
)

_EI_SCHEMA = '{"importance": 0.0, "reasoning": "string"}'


def enrich_edge_importance(edges: list[Edge], client: LLMClient) -> list[Edge]:
    """Ask LLM to rate each edge's importance to the domain."""
    def process(edge: Edge) -> None:
        prompt = (
            f"Fact: {edge.source} {edge.relation.replace('_', ' ')} {edge.target}\n"
            f"Evidence: {edge.evidence_text}\n"
            f"Context: This edge connects entities of type "
            f"{edge.source_type} and {edge.target_type}."
        )
        try:
            result = client.chat_json(_EI_SYSTEM, prompt, _EI_SCHEMA)
            importance = float(result.get("importance", 0.5))
            edge.importance = max(0.0, min(1.0, importance))
        except Exception as exc:
            logger.warning("Importance enrichment failed for edge %s: %s", edge.id, exc)
    _run_parallel(edges, process, client)
    return edges


def enrich_graph(
    entities: list[Entity],
    edges: list[Edge],
    client: LLMClient,
    steps: list[str] | None = None,
) -> tuple[list[Entity], list[Edge]]:
    """Run selected LLM enrichment steps on the graph.

    Steps: "questions", "summaries", "contradictions", "entities",
           "infer", "importance". Default: all.
    """
    all_steps = {"questions", "summaries", "contradictions", "entities", "infer", "importance"}
    active = set(steps) & all_steps if steps else all_steps

    if "questions" in active:
        edges = enrich_edge_questions(edges, client)
    if "summaries" in active:
        edges = enrich_edge_summaries(edges, client)
    if "importance" in active:
        edges = enrich_edge_importance(edges, client)
    if "contradictions" in active:
        edges = enrich_contradiction_analysis(edges, client)
    if "entities" in active:
        entities = enrich_entities(entities, edges, client)
    if "infer" in active:
        edges = enrich_inference_edges(edges, client)

    return entities, edges


def finalize_enriched_index(
    source_index: Path,
    output_index: Path,
    entities: list[Entity],
    edges: list[Edge],
) -> dict[str, int]:
    """Persist a recalibrated enriched index derived from a source index."""
    entities, edges = resolve_entity_aliases(entities, edges)
    _recompute_graph_stats(entities, edges)
    calibrate_edges(edges)
    communities, reports = build_communities(entities, edges)
    if output_index.exists():
        shutil.rmtree(output_index)
    shutil.copytree(source_index, output_index)
    tables = {
        "entities": entities,
        "calibrated_edges": edges,
        "communities": communities,
        "community_reports": reports,
    }
    for name, values in tables.items():
        export_table(output_index, name, [asdict(value) for value in values])
    manifest_path = output_index / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["enriched"] = True
    manifest["counts"] = {
        **manifest.get("counts", {}),
        **{name: len(values) for name, values in tables.items()},
    }
    write_json(manifest_path, manifest)
    return {name: len(values) for name, values in tables.items()}


def resolve_entity_aliases(
    entities: list[Entity],
    edges: list[Edge],
) -> tuple[list[Entity], list[Edge]]:
    """Merge entity aliases and deduplicate edges after enrichment."""
    alias_to_title: dict[str, str] = {}
    for entity in sorted(entities, key=lambda item: item.degree, reverse=True):
        alias_to_title.setdefault(_entity_key(entity.title), entity.title)
        for alias in entity.aliases:
            alias_to_title.setdefault(_entity_key(alias), entity.title)

    for edge in edges:
        edge.source = alias_to_title.get(_entity_key(edge.source), edge.source)
        edge.target = alias_to_title.get(_entity_key(edge.target), edge.target)

    merged: dict[str, Entity] = {}
    for entity in entities:
        canonical_title = alias_to_title.get(_entity_key(entity.title), entity.title)
        key = _entity_key(canonical_title)
        target = merged.get(key)
        if target is None:
            entity.title = canonical_title
            merged[key] = entity
            continue
        target.aliases = sorted(set(target.aliases + entity.aliases + [entity.title]))
        target.text_unit_ids = sorted(set(target.text_unit_ids + entity.text_unit_ids))
        target.degree += entity.degree
        if not target.enriched_description:
            target.enriched_description = entity.enriched_description

    deduped_edges: dict[tuple[str, str, str], Edge] = {}
    for edge in edges:
        if _entity_key(edge.source) == _entity_key(edge.target):
            continue
        key = (_entity_key(edge.source), _entity_key(edge.target), edge.relation)
        existing = deduped_edges.get(key)
        if existing is None:
            deduped_edges[key] = edge
            continue
        existing.source_docs = sorted(set(existing.source_docs + edge.source_docs))
        existing.text_unit_ids = sorted(set(existing.text_unit_ids + edge.text_unit_ids))
        existing.weight += edge.weight
    return list(merged.values()), list(deduped_edges.values())


def _recompute_graph_stats(entities: list[Entity], edges: list[Edge]) -> None:
    """Recompute entity degrees and edge combined degrees in place."""
    degree: defaultdict[str, int] = defaultdict(int)
    for edge in edges:
        degree[edge.source] += 1
        degree[edge.target] += 1
    for entity in entities:
        entity.degree = degree[entity.title]
    for edge in edges:
        edge.combined_degree = degree[edge.source] + degree[edge.target]


def _entity_key(value: str) -> str:
    return re.sub(r"[^\w]+", "", value.casefold())


def _run_parallel(items: list[Any], function: Any, client: LLMClient) -> list[Any]:
    """Run enrichment tasks with the worker count from the LLM client config."""
    if not items:
        return []
    workers = max(1, getattr(getattr(client, "config", None), "workers", 12))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm-enrich") as executor:
        return list(executor.map(function, items))
