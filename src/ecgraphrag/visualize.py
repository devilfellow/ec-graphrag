from __future__ import annotations

import argparse
import json
import html
from pathlib import Path
from typing import Any

from .models import Edge, Entity
from .storage import read_jsonl, write_json


def build_networkx_graph(index_path: Path):
    """Build a NetworkX multigraph from persisted index tables."""
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Graph visualization/export requires networkx") from exc
    entities = [Entity(**row) for row in read_jsonl(index_path / "entities.jsonl")]
    edges = [Edge(**row) for row in read_jsonl(index_path / "calibrated_edges.jsonl")]
    graph = nx.MultiGraph()
    for entity in entities:
        graph.add_node(
            entity.title,
            id=entity.id,
            type=entity.type,
            description=entity.description,
            enriched_description=entity.enriched_description,
            aliases=", ".join(entity.aliases),
            category=entity.category,
            degree=entity.degree,
        )
    for edge in edges:
        graph.add_edge(
            edge.source,
            edge.target,
            key=edge.id,
            id=edge.id,
            relation=edge.relation,
            description=edge.description,
            semantic_summary=edge.semantic_summary,
            generated_questions=" | ".join(edge.generated_questions),
            evidence_type=edge.evidence_type,
            evidence_text=edge.evidence_text,
            importance=edge.importance,
            weight=edge.weight,
            reliability=edge.reliability,
            evidence_score=edge.evidence_score,
            ontology_score=edge.ontology_score,
            structural_score=edge.structural_score,
            consistency_score=edge.consistency_score,
            text_unit_ids=",".join(edge.text_unit_ids),
        )
    return graph


def export_graph(index_path: Path, output_dir: Path | None = None, max_nodes: int = 200) -> dict[str, Any]:
    """Export graph artifacts to GraphML, HTML, PNG, and a manifest."""
    output_dir = output_dir or (index_path / "graph_exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = build_networkx_graph(index_path)
    result: dict[str, Any] = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "output_dir": str(output_dir),
    }
    try:
        import networkx as nx
        graphml_path = output_dir / "graph.graphml"
        nx.write_graphml(graph, graphml_path)
        result["graphml"] = str(graphml_path)
    except Exception as exc:  # pragma: no cover
        result["graphml_error"] = str(exc)

    html_path = output_dir / "enriched_graph.html"
    png_path = output_dir / "graph.png"
    try:
        _write_pyvis_html(graph, html_path, max_nodes=max_nodes)
        result["html"] = str(html_path)
        result["html_backend"] = "pyvis"
    except Exception as exc:
        result["pyvis_error"] = str(exc)
        try:
            _write_vis_network_html(graph, html_path, max_nodes=max_nodes)
            result["html"] = str(html_path)
            result["html_backend"] = "vis-network"
        except Exception as fallback_exc:
            result["html_error"] = str(fallback_exc)
    try:
        _write_png(graph, png_path, max_nodes=max_nodes)
        result["png"] = str(png_path)
    except Exception as exc:
        result["png_error"] = str(exc)
    write_json(output_dir / "graph_export_manifest.json", result)
    return result


def _subgraph_top(graph, max_nodes: int):
    if graph.number_of_nodes() <= max_nodes:
        return graph
    degree = dict(graph.degree())
    selected = sorted(graph.nodes(), key=lambda n: degree[n], reverse=True)[:max_nodes]
    return graph.subgraph(selected).copy()


def _write_pyvis_html(graph, path: Path, max_nodes: int = 200) -> None:
    """Write an interactive graph HTML file using PyVis."""
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise RuntimeError("Install pyvis for interactive HTML visualization") from exc
    subgraph = _subgraph_top(graph, max_nodes)
    net = Network(height="800px", width="100%", notebook=False, directed=False, bgcolor="#ffffff")
    for node, data in subgraph.nodes(data=True):
        description = data.get("enriched_description") or data.get("description") or node
        aliases = data.get("aliases") or ""
        category = data.get("category") or data.get("type") or "Entity"
        title = f"{description}<br>category={category}<br>aliases={aliases}"
        enriched = bool(data.get("enriched_description") or aliases or data.get("category"))
        net.add_node(
            node,
            label=str(node)[:42],
            title=title,
            value=max(1, int(data.get("degree") or 1)),
            color="#7fc8f8" if enriched else "#d9e2ec",
        )
    for source, target, data in subgraph.edges(data=True):
        reliability = float(data.get("reliability") or 0.0)
        importance = float(data.get("importance") or 0.0)
        evidence_type = str(data.get("evidence_type") or "")
        summary = data.get("semantic_summary") or data.get("description") or ""
        questions = data.get("generated_questions") or ""
        title = (
            f"{data.get('relation')} | reliability={reliability:.3f} | importance={importance:.3f}"
            f"<br>evidence_type={evidence_type}<br>{summary}<br>questions={questions}"
        )
        net.add_edge(
            source,
            target,
            title=title,
            value=max(1, reliability * 5),
            label=str(data.get("relation", ""))[:18],
            color="#f59e0b" if evidence_type == "inferred" else "#64748b",
            dashes=evidence_type == "inferred",
        )
    net.repulsion(node_distance=120, central_gravity=0.2, spring_length=120, spring_strength=0.05)
    net.write_html(str(path), notebook=False, open_browser=False)


def _write_png(graph, path: Path, max_nodes: int = 200) -> None:
    """Write a static PNG preview of the graph."""
    import matplotlib.pyplot as plt
    import networkx as nx
    subgraph = _subgraph_top(graph, max_nodes)
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(subgraph, seed=42, k=0.35)
    reliabilities = [float(data.get("reliability") or 0.2) for _, _, data in subgraph.edges(data=True)]
    widths = [0.5 + 2.5 * value for value in reliabilities]
    nx.draw_networkx_nodes(subgraph, pos, node_size=140, alpha=0.85)
    nx.draw_networkx_edges(subgraph, pos, width=widths, alpha=0.35)
    nx.draw_networkx_labels(subgraph, pos, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _write_vis_network_html(graph, path: Path, max_nodes: int = 200) -> None:
    """Write an interactive graph HTML file using bundled vis-network assets."""
    subgraph = _subgraph_top(graph, max_nodes)
    project_root = Path(__file__).resolve().parents[2]
    asset_dir = project_root / "notebooks" / "lib" / "vis-9.1.2"
    js = (asset_dir / "vis-network.min.js").read_text(encoding="utf-8")
    css = (asset_dir / "vis-network.css").read_text(encoding="utf-8")
    nodes = []
    for node, data in subgraph.nodes(data=True):
        description = data.get("enriched_description") or data.get("description") or node
        aliases = data.get("aliases") or ""
        category = data.get("category") or data.get("type") or "Entity"
        enriched = bool(data.get("enriched_description") or aliases or data.get("category"))
        nodes.append({
            "id": node,
            "label": str(node)[:42],
            "title": f"{html.escape(str(description))}<br>category={html.escape(str(category))}<br>aliases={html.escape(str(aliases))}",
            "value": max(1, int(data.get("degree") or 1)),
            "color": "#7fc8f8" if enriched else "#d9e2ec",
        })
    edges = []
    for source, target, data in subgraph.edges(data=True):
        reliability = float(data.get("reliability") or 0.0)
        importance = float(data.get("importance") or 0.0)
        evidence_type = str(data.get("evidence_type") or "")
        summary = data.get("semantic_summary") or data.get("description") or ""
        questions = data.get("generated_questions") or ""
        edges.append({
            "from": source,
            "to": target,
            "label": str(data.get("relation", ""))[:18],
            "title": (
                f"{html.escape(str(data.get('relation', '')))} | reliability={reliability:.3f} | "
                f"importance={importance:.3f}<br>evidence_type={html.escape(evidence_type)}"
                f"<br>{html.escape(str(summary))}<br>questions={html.escape(str(questions))}"
            ),
            "value": max(1, reliability * 5),
            "color": "#f59e0b" if evidence_type == "inferred" else "#64748b",
            "dashes": evidence_type == "inferred",
        })
    nodes_json = json.dumps(nodes, ensure_ascii=False).replace("</", "<\\/")
    edges_json = json.dumps(edges, ensure_ascii=False).replace("</", "<\\/")
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>{css}
html, body, #network {{ width: 100%; height: 100%; margin: 0; }}
</style>
<script>{js}</script>
</head>
<body>
<div id="network"></div>
<script>
const nodes = new vis.DataSet({nodes_json});
const edges = new vis.DataSet({edges_json});
const options = {{
  interaction: {{ hover: true, navigationButtons: true }},
  physics: {{ solver: "repulsion", repulsion: {{ nodeDistance: 120 }} }},
  edges: {{ smooth: true, font: {{ size: 10 }} }}
}};
new vis.Network(document.getElementById("network"), {{ nodes, edges }}, options);
</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    """Run graph export from the CLI."""
    parser = argparse.ArgumentParser(prog="ecgraphrag.visualize")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-nodes", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(export_graph(args.index, args.output, args.max_nodes), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
