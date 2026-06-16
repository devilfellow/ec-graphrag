from __future__ import annotations

import math
from collections import Counter, defaultdict

from .models import Edge
from .text import content_tokens


def calibrate_edges(
    edges: list[Edge],
    weights: dict[str, float] | None = None,
) -> list[Edge]:
    """Compute evidence, ontology, structural, consistency, and reliability scores."""
    coefficients = weights or {
        "evidence": 0.35,
        "ontology": 0.25,
        "structural": 0.20,
        "consistency": 0.20,
    }
    total = sum(coefficients.values())
    coefficients = {key: value / total for key, value in coefficients.items()}
    relation_counts = Counter(edge.relation for edge in edges)
    pairs: defaultdict[tuple[str, str], list[Edge]] = defaultdict(list)
    max_degree = max((edge.combined_degree for edge in edges), default=1)
    for edge in edges:
        pairs[(edge.source.casefold(), edge.target.casefold())].append(edge)

    for edge in edges:
        aliases = content_tokens(f"{edge.source} {edge.target} {edge.relation}")
        evidence_tokens = set(content_tokens(edge.evidence_text))
        coverage = sum(token in evidence_tokens for token in set(aliases)) / max(1, len(set(aliases)))
        explicitness = {"explicit": 1.0, "table": 0.7, "implicit": 0.4}.get(edge.evidence_type, 0.4)
        redundancy = min(1.0, 0.5 + 0.2 * max(0, len(edge.source_docs) - 1))
        edge.evidence_score = _clip(0.4 * edge.llm_conf + 0.3 * coverage + 0.2 * explicitness + 0.1 * redundancy)
        edge.ontology_score = 1.0 if edge.ontology_ok else 0.0
        frequency = math.log1p(relation_counts[edge.relation]) / math.log1p(max(1, len(edges)))
        centrality = math.log1p(edge.combined_degree) / math.log1p(max_degree)
        edge.structural_score = _clip(0.55 * centrality + 0.45 * frequency)
        conflicts = [item for item in pairs[(edge.source.casefold(), edge.target.casefold())] if item.relation != edge.relation]
        if edge.conflict_status != "contradiction":
            edge.conflict_status = "review" if conflicts else "ok"
        contradiction_penalty = 0.5 if edge.conflict_status == "contradiction" else 0.0
        edge.consistency_score = _clip(1.0 - contradiction_penalty - min(0.75, 0.25 * len(conflicts)))
        edge.reliability = _clip(
            coefficients["evidence"] * edge.evidence_score
            + coefficients["ontology"] * edge.ontology_score
            + coefficients["structural"] * edge.structural_score
            + coefficients["consistency"] * edge.consistency_score
        )
    return edges


def _clip(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)
