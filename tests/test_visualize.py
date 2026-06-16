from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from ecgraphrag.models import Edge, Entity
from ecgraphrag.storage import export_table
from ecgraphrag.visualize import export_graph


class VisualizeTest(unittest.TestCase):
    def test_export_graph_falls_back_to_vis_network_html(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = root / "index"
            output = root / "output"
            index.mkdir()
            entities = [Entity("a", "Alpha"), Entity("b", "Beta")]
            edges = [
                Edge(
                    "edge", "Alpha", "Beta", "associated_with", "Alpha relates to Beta",
                    edge_text="Alpha relates to Beta", reliability=0.8,
                )
            ]
            export_table(index, "entities", [asdict(item) for item in entities])
            export_table(index, "calibrated_edges", [asdict(item) for item in edges])
            with patch("ecgraphrag.visualize._write_pyvis_html", side_effect=RuntimeError("missing pyvis")):
                result = export_graph(index, output, max_nodes=10)
            self.assertEqual(result["html_backend"], "vis-network")
            self.assertTrue(Path(result["html"]).exists())
            self.assertIn("missing pyvis", result["pyvis_error"])


if __name__ == "__main__":
    unittest.main()
