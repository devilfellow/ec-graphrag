from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    """Normalized source document used for indexing and evaluation."""

    id: str
    title: str
    text: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextUnit:
    """Overlapping text chunk derived from a document."""

    id: str
    document_id: str
    text: str
    position: int
    token_count: int


@dataclass
class Entity:
    """Canonical graph node with extraction and enrichment metadata."""

    id: str
    title: str
    type: str = "Entity"
    description: str = ""
    text_unit_ids: list[str] = field(default_factory=list)
    degree: int = 0
    enriched_description: str = ""
    aliases: list[str] = field(default_factory=list)
    category: str = ""


@dataclass
class Edge:
    """Knowledge graph relationship with evidence, calibration, and retrieval fields."""

    id: str
    source: str
    target: str
    relation: str
    description: str
    weight: float = 1.0
    text_unit_ids: list[str] = field(default_factory=list)
    evidence_text: str = ""
    evidence_type: str = "implicit"
    source_docs: list[str] = field(default_factory=list)
    qualifiers: dict[str, Any] = field(default_factory=dict)
    source_type: str = "Entity"
    target_type: str = "Entity"
    ontology_ok: bool = True
    conflict_status: str = "ok"
    llm_conf: float = 0.5
    evidence_score: float = 0.0
    ontology_score: float = 0.0
    structural_score: float = 0.0
    consistency_score: float = 0.0
    reliability: float = 0.0
    edge_text: str = ""
    edge_emb_id: str = ""
    useful_for: list[str] = field(default_factory=list)
    retrieval_stats: dict[str, int] = field(
        default_factory=lambda: {"retrieved": 0, "used": 0}
    )
    valid_from: str | None = None
    valid_to: str | None = None
    combined_degree: int = 0
    generated_questions: list[str] = field(default_factory=list)
    semantic_summary: str = ""
    contradiction_info: str = ""
    importance: float = 0.5


@dataclass
class Community:
    """Group of entities and edges found by graph community detection."""

    id: str
    entity_ids: list[str]
    edge_ids: list[str]
    level: int = 0


@dataclass
class CommunityReport:
    """Retrieval-oriented summary for a graph community."""

    id: str
    community_id: str
    title: str
    summary: str
    entity_ids: list[str]
    edge_ids: list[str]
    reliability: float


@dataclass
class Candidate:
    """Ranked retrieval context item returned by the retriever."""

    id: str
    kind: str
    text: str
    reliability: float
    utility: float
    score: float
    token_count: int
    source_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def to_dict(value: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a dictionary."""
    return asdict(value)
