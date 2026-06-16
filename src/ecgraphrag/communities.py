from __future__ import annotations

from collections import defaultdict

from .models import Community, CommunityReport, Edge, Entity
from .text import stable_id


def build_communities(
    entities: list[Entity], edges: list[Edge]
) -> tuple[list[Community], list[CommunityReport]]:
    """Build graph communities and retrieval reports from calibrated edges."""
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    edge_lookup: defaultdict[frozenset[str], list[Edge]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
        edge_lookup[frozenset((edge.source, edge.target))].append(edge)

    try:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(entity.title for entity in entities)
        graph.add_weighted_edges_from(
            (edge.source, edge.target, max(0.01, edge.reliability))
            for edge in edges
        )
        member_groups = [
            sorted(group)
            for group in nx.community.greedy_modularity_communities(graph, weight="weight")
        ]
    except (ImportError, ValueError, ZeroDivisionError):
        member_groups = [[entity.title] for entity in entities]

    communities: list[Community] = []
    reports: list[CommunityReport] = []
    for members in member_groups:
        component_edges: dict[str, Edge] = {}
        member_set = set(members)
        for current in members:
            for neighbor in adjacency[current]:
                if neighbor not in member_set:
                    continue
                for edge in edge_lookup[frozenset((current, neighbor))]:
                    component_edges[edge.id] = edge
        community_id = stable_id("community", *sorted(members))
        edge_ids = sorted(component_edges)
        communities.append(Community(community_id, sorted(members), edge_ids))
        ranked = sorted(component_edges.values(), key=lambda edge: (-edge.reliability, -edge.weight))
        report_edges = ranked[:8]
        facts = [edge.semantic_summary or edge.description for edge in report_edges]
        summary = " ".join(facts) if facts else f"Entity: {members[0]}."
        reliability = sum(edge.reliability for edge in ranked) / len(ranked) if ranked else 0.5
        reports.append(CommunityReport(
            id=stable_id("report", community_id),
            community_id=community_id,
            title=", ".join(sorted(members)[:4]),
            summary=summary,
            entity_ids=sorted(members),
            edge_ids=[edge.id for edge in report_edges],
            reliability=round(reliability, 6),
        ))
    return communities, reports
