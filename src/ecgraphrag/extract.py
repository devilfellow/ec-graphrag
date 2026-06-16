from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Document, Edge, Entity, TextUnit
from .openrouter import OpenRouterClient
from .text import sentences, stable_id

logger = logging.getLogger(__name__)

ENTITY_RE = re.compile(
    r"\b(?:[A-ZА-ЯЁ][\w-]*(?:\s+[A-ZА-ЯЁ][\w-]*){0,4}|[A-ZА-ЯЁ]{2,}(?:[-_][A-ZА-ЯЁ0-9]+)*)\b",
    re.UNICODE,
)
RELATIONS = [
    ("causes", re.compile(r"\b(?:causes?|caused|leads? to|results? in|вызывает?|приводит к|стал[аои]? причиной)\b", re.I)),
    ("prevents", re.compile(r"\b(?:prevents?|blocks?|мешает|предотвращает|блокирует)\b", re.I)),
    ("depends_on", re.compile(r"\b(?:depends? on|requires?|зависит от|требует)\b", re.I)),
    ("part_of", re.compile(r"\b(?:part of|belongs? to|часть|входит в)\b", re.I)),
    ("located_in", re.compile(r"\b(?:located in|based in|находится в|расположен[ао]? в)\b", re.I)),
    ("associated_with", re.compile(r"\b(?:associated with|related to|связан[ао]? с|относится к)\b", re.I)),
]
RELATION_ONTOLOGY = {
    "acquired_by", "associated_with", "causes", "compares_to", "competes_with",
    "contributes_to", "depends_on", "featured_in", "hosts", "impacted_by",
    "indirectly_causes", "located_in", "member_of", "part_of", "plays_for", "prevents", "produces",
    "reports_on", "uses", "works_for",
}
RELATION_ALIASES = {
    "competes_against": "competes_with",
    "related_to": "associated_with",
    "played_against": "competes_with",
    "has_player": "plays_for",
    "scores": "contributes_to",
}


def _entities(text: str) -> list[str]:
    result = []
    for match in ENTITY_RE.finditer(text):
        value = match.group(0).strip()
        if len(value) > 1 and value not in result:
            result.append(value)
    return result


def _relation(sentence: str) -> tuple[str, str]:
    for name, pattern in RELATIONS:
        match = pattern.search(sentence)
        if match:
            return name, "explicit"
    return "co_occurs", "implicit"


def extract_graph(
    documents: list[Document],
    units: list[TextUnit],
    extractor: str = "rules",
    llm_client: OpenRouterClient | None = None,
    max_llm_units: int | None = None,
    llm_cache_dir: Path | None = None,
    resume: bool = True,
    continue_on_error: bool = True,
    extraction_stats: dict[str, int] | None = None,
) -> tuple[list[Entity], list[Edge]]:
    """Extract graph entities and edges from text units with rules or an LLM."""
    if extractor == "llm":
        return _extract_graph_llm(
            documents,
            units,
            llm_client=llm_client,
            max_llm_units=max_llm_units,
            cache_dir=llm_cache_dir,
            resume=resume,
            continue_on_error=continue_on_error,
            stats=extraction_stats,
        )

    docs = {document.id: document for document in documents}
    entities: dict[str, Entity] = {}
    edges: dict[tuple[str, str, str], Edge] = {}
    mentions: defaultdict[str, set[str]] = defaultdict(set)
    descriptions: defaultdict[str, list[str]] = defaultdict(list)

    for unit in units:
        document = docs[unit.document_id]
        supplied = document.metadata.get("relationships", [])
        sentence_rows: list[tuple[str, str, str, str]] = []
        for item in supplied if isinstance(supplied, list) else []:
            sentence_rows.append((
                str(item["source"]),
                str(item["target"]),
                str(item.get("relation", "related_to")),
                str(item.get("description") or unit.text),
            ))
        for sentence in sentences(unit.text):
            names = _entities(sentence)
            if len(names) >= 2:
                relation, _ = _relation(sentence)
                sentence_rows.append((names[0], names[1], relation, sentence))

        for source, target, relation, evidence in sentence_rows:
            if source == target:
                continue
            raw_relation = relation
            relation, ontology_ok = _normalize_relation(relation)
            evidence_type = _relation(evidence)[1] if raw_relation == "co_occurs" else "explicit"
            for name in (source, target):
                entity_id = stable_id("ent", name.casefold())
                mentions[name].add(unit.id)
                descriptions[name].append(evidence)
                entities.setdefault(entity_id, Entity(entity_id, name))
            key = (source.casefold(), target.casefold(), relation)
            if key not in edges:
                edge_id = stable_id("edge", *key)
                edges[key] = Edge(
                    id=edge_id,
                    source=source,
                    target=target,
                    relation=relation,
                    description=evidence,
                    evidence_text=evidence,
                    evidence_type=evidence_type,
                    source_docs=[document.id],
                    text_unit_ids=[unit.id],
                    llm_conf=0.85 if supplied else (0.75 if evidence_type == "explicit" else 0.45),
                    ontology_ok=ontology_ok,
                    edge_text=f"{source} {relation.replace('_', ' ')} {target}. {evidence}",
                    edge_emb_id=f"emb_{edge_id}",
                    useful_for=_useful_for(relation),
                )
            else:
                edge = edges[key]
                edge.weight += 1.0
                if unit.id not in edge.text_unit_ids:
                    edge.text_unit_ids.append(unit.id)
                if document.id not in edge.source_docs:
                    edge.source_docs.append(document.id)

    result_entities = list(entities.values())
    degree: defaultdict[str, int] = defaultdict(int)
    for edge in edges.values():
        degree[edge.source] += 1
        degree[edge.target] += 1
    for entity in result_entities:
        entity.text_unit_ids = sorted(mentions[entity.title])
        entity.degree = degree[entity.title]
        entity.description = descriptions[entity.title][0] if descriptions[entity.title] else ""
    for edge in edges.values():
        edge.combined_degree = degree[edge.source] + degree[edge.target]
    return result_entities, list(edges.values())


def _useful_for(relation: str) -> list[str]:
    if relation in {"causes", "prevents", "depends_on"}:
        return ["causal", "diagnostic"]
    if relation in {"located_in", "part_of"}:
        return ["factual", "structural"]
    return ["factual", "exploratory"]



SYSTEM_PROMPT = """You extract a compact knowledge graph for GraphRAG.
Return ONLY valid JSON with keys `entities` and `relationships`.
Use concise canonical entity names and reuse exactly the same name for the same entity.
Include only facts directly supported by the supplied chunk. Never add outside knowledge.
For each relationship include source, target, relation, description, evidence_text,
evidence_type, source_type, target_type, llm_conf, useful_for, qualifiers.
`evidence_text` must be a short verbatim span from the supplied chunk.
Use one relation from this ontology: acquired_by, associated_with, causes, compares_to,
competes_with, contributes_to, depends_on, featured_in, hosts, impacted_by, located_in,
indirectly_causes, member_of, part_of, plays_for, prevents, produces, reports_on, uses, works_for.
`useful_for` should contain query intents: factual, causal, diagnostic, exploratory, global.
"""

SCHEMA_HINT = """{
  "entities": [{"name": "string", "type": "string", "description": "string"}],
  "relationships": [{
    "source": "string", "target": "string", "relation": "string",
    "description": "string", "evidence_text": "string",
    "evidence_type": "explicit|implicit|table", "source_type": "string",
    "target_type": "string", "llm_conf": 0.0,
    "useful_for": ["factual"], "qualifiers": {"time_period": null, "location": null, "condition": null}
  }]
}"""


def _extract_graph_llm(
    documents: list[Document],
    units: list[TextUnit],
    llm_client: OpenRouterClient | None = None,
    max_llm_units: int | None = None,
    cache_dir: Path | None = None,
    resume: bool = True,
    continue_on_error: bool = True,
    stats: dict[str, int] | None = None,
) -> tuple[list[Entity], list[Edge]]:
    """Extract and merge graph facts from LLM JSON responses with cache/resume support."""
    docs = {document.id: document for document in documents}
    client = llm_client or OpenRouterClient()
    entities: dict[str, Entity] = {}
    edges: dict[tuple[str, str, str], Edge] = {}
    mentions: defaultdict[str, set[str]] = defaultdict(set)
    descriptions: defaultdict[str, list[str]] = defaultdict(list)
    canonical_names: dict[str, str] = {}
    selected_units = units[:max_llm_units] if max_llm_units else units
    stats = stats if stats is not None else {}
    stats.update({"successful_units": 0, "failed_units": 0, "cached_units": 0})
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def load_unit(unit: TextUnit) -> tuple[TextUnit, dict[str, Any] | None, bool, Exception | None]:
        document = docs[unit.document_id]
        cache_path = cache_dir / f"{unit.id}.json" if cache_dir else None
        try:
            payload = None
            loaded_from_cache = False
            if resume and cache_path and cache_path.exists():
                try:
                    payload = _validate_extraction_payload(
                        json.loads(cache_path.read_text(encoding="utf-8"))
                    )
                    loaded_from_cache = True
                except (json.JSONDecodeError, ValueError):
                    cache_path.unlink()
            if payload is None:
                payload = _validate_extraction_payload(client.chat_json(
                    SYSTEM_PROMPT,
                    f"Document title: {document.title}\nChunk id: {unit.id}\nText:\n{unit.text}",
                    SCHEMA_HINT,
                ))
                if cache_path:
                    cache_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            return unit, payload, loaded_from_cache, None
        except Exception as exc:
            return unit, None, False, exc

    workers = max(1, getattr(getattr(client, "config", None), "workers", 12))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm-extract") as executor:
        results = list(executor.map(load_unit, selected_units))

    for position, (unit, payload, cached, error) in enumerate(results, start=1):
        document = docs[unit.document_id]
        if error is None and payload is not None:
            stats["successful_units"] += 1
            stats["cached_units"] += int(cached)
            _clear_extraction_error(cache_dir, unit.id)
            _merge_llm_payload(
                payload, unit, document, entities, edges, mentions, descriptions, canonical_names
            )
        else:
            stats["failed_units"] += 1
            _write_extraction_error(cache_dir, unit, document, error or RuntimeError("Unknown extraction error"))
            if not continue_on_error:
                raise error or RuntimeError("Unknown extraction error")
        logger.info(
            f"LLM extraction {position}/{len(selected_units)}: "
            f"successful={stats['successful_units']} cached={stats['cached_units']} "
            f"failed={stats['failed_units']}"
        )
    return _finalize_entities_edges(list(entities.values()), list(edges.values()), mentions, descriptions)


def _merge_llm_payload(
    payload: dict[str, Any],
    unit: TextUnit,
    document: Document,
    entities: dict[str, Entity],
    edges: dict[tuple[str, str, str], Edge],
    mentions: defaultdict[str, set[str]],
    descriptions: defaultdict[str, list[str]],
    canonical_names: dict[str, str],
) -> None:
    """Merge a validated extraction payload into canonical entity and edge maps."""
    for item in payload["entities"]:
        if not isinstance(item, dict):
            continue
        name = _canonical_entity_name(
            str(item.get("name") or item.get("title") or "").strip(),
            canonical_names,
        )
        if not name:
            continue
        entity_id = stable_id("ent", name.casefold())
        entity = entities.setdefault(entity_id, Entity(entity_id, name))
        entity.type = str(item.get("type") or entity.type or "Entity")
        desc = str(item.get("description") or "").strip()
        if desc:
            descriptions[name].append(desc)
        mentions[name].add(unit.id)
    for item in payload["relationships"]:
        if not isinstance(item, dict):
            continue
        source = _canonical_entity_name(str(item.get("source") or "").strip(), canonical_names)
        target = _canonical_entity_name(str(item.get("target") or "").strip(), canonical_names)
        raw_relation = str(item.get("relation") or "associated_with").strip()
        relation, ontology_ok = _normalize_relation(raw_relation)
        if not source or not target or source == target:
            continue
        description = str(item.get("description") or f"{source} {relation.replace('_', ' ')} {target}").strip()
        evidence = str(item.get("evidence_text") or description).strip()
        if evidence.casefold() not in unit.text.casefold():
            evidence = description if description.casefold() in unit.text.casefold() else unit.text
        source_type = str(item.get("source_type") or "Entity")
        target_type = str(item.get("target_type") or "Entity")
        useful_for = item.get("useful_for") if isinstance(item.get("useful_for"), list) else _useful_for(relation)
        qualifiers = item.get("qualifiers") if isinstance(item.get("qualifiers"), dict) else {}
        try:
            llm_conf = float(item.get("llm_conf") or item.get("confidence") or 0.75)
        except (TypeError, ValueError):
            llm_conf = 0.75
        evidence_type = str(item.get("evidence_type") or "explicit")
        for name, typ in ((source, source_type), (target, target_type)):
            entity_id = stable_id("ent", name.casefold())
            entity = entities.setdefault(entity_id, Entity(entity_id, name, type=typ))
            if entity.type == "Entity" and typ:
                entity.type = typ
            mentions[name].add(unit.id)
            descriptions[name].append(evidence)
        key = (source.casefold(), target.casefold(), relation)
        if key not in edges:
            edge_id = stable_id("edge", *key)
            edges[key] = Edge(
                id=edge_id,
                source=source,
                target=target,
                relation=relation,
                description=description,
                evidence_text=evidence,
                evidence_type=evidence_type,
                source_docs=[document.id],
                text_unit_ids=[unit.id],
                qualifiers=qualifiers,
                source_type=source_type,
                target_type=target_type,
                ontology_ok=ontology_ok,
                llm_conf=max(0.0, min(1.0, llm_conf)),
                edge_text=f"{source} {relation.replace('_', ' ')} {target}. {description}",
                edge_emb_id=f"emb_{edge_id}",
                useful_for=[str(tag) for tag in useful_for],
            )
        else:
            edge = edges[key]
            edge.weight += 1.0
            if unit.id not in edge.text_unit_ids:
                edge.text_unit_ids.append(unit.id)
            if document.id not in edge.source_docs:
                edge.source_docs.append(document.id)


def _validate_extraction_payload(payload: Any) -> dict[str, list[dict[str, Any]]]:
    """Validate an LLM extraction response and drop malformed individual rows."""
    if not isinstance(payload, dict):
        raise ValueError("Extraction response must be a JSON object")
    if "entities" not in payload or "relationships" not in payload:
        raise ValueError("Extraction response must contain entities and relationships")
    if not isinstance(payload["entities"], list) or not isinstance(payload["relationships"], list):
        raise ValueError("Extraction entities and relationships must be arrays")
    entities = [
        item for item in payload["entities"]
        if isinstance(item, dict) and str(item.get("name") or item.get("title") or "").strip()
    ]
    relationships = [
        item for item in payload["relationships"]
        if isinstance(item, dict)
        and str(item.get("source") or "").strip()
        and str(item.get("target") or "").strip()
    ]
    if not entities and not relationships:
        raise ValueError("Extraction response contains no valid entities or relationships")
    return {"entities": entities, "relationships": relationships}


def _write_extraction_error(
    cache_dir: Path | None,
    unit: TextUnit,
    document: Document,
    exc: Exception,
) -> None:
    """Record a final extraction failure for a text unit."""
    if not cache_dir:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text_unit_id": unit.id,
        "document_id": document.id,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    _clear_extraction_error(cache_dir, unit.id)
    with (cache_dir.parent / "llm_errors.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clear_extraction_error(cache_dir: Path | None, text_unit_id: str) -> None:
    """Remove resolved errors for a text unit from the extraction error log."""
    if not cache_dir:
        return
    path = cache_dir.parent / "llm_errors.jsonl"
    if not path.exists():
        return
    unresolved = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("text_unit_id") != text_unit_id:
            unresolved.append(record)
    with path.open("w", encoding="utf-8") as stream:
        for record in unresolved:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _normalize_relation(value: str) -> tuple[str, bool]:
    """Normalize relation labels to the supported ontology."""
    relation = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    relation = RELATION_ALIASES.get(relation, relation)
    if relation in RELATION_ONTOLOGY:
        return relation, True
    return "associated_with", False


def _canonical_entity_name(value: str, canonical_names: dict[str, str]) -> str:
    key = re.sub(r"[^\w]+", "", value.casefold())
    if not key:
        return ""
    canonical_names.setdefault(key, value)
    return canonical_names[key]


def _finalize_entities_edges(
    result_entities: list[Entity],
    result_edges: list[Edge],
    mentions: defaultdict[str, set[str]],
    descriptions: defaultdict[str, list[str]],
) -> tuple[list[Entity], list[Edge]]:
    """Populate graph degrees and entity descriptions after extraction."""
    degree: defaultdict[str, int] = defaultdict(int)
    for edge in result_edges:
        degree[edge.source] += 1
        degree[edge.target] += 1
    for entity in result_entities:
        entity.text_unit_ids = sorted(mentions[entity.title])
        entity.degree = degree[entity.title]
        entity.description = descriptions[entity.title][0] if descriptions[entity.title] else entity.description
    for edge in result_edges:
        edge.combined_degree = degree[edge.source] + degree[edge.target]
    return result_entities, result_edges
