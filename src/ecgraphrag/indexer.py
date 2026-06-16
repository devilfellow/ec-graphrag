from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .calibrate import calibrate_edges
from .communities import build_communities
from .extract import extract_graph
from .ingest import chunk_documents, ingest
from .openrouter import OpenRouterClient, OpenRouterConfig
from .storage import export_table, write_json


class GraphRAGIndexer:
    """Build a persisted GraphRAG index from documents."""

    def __init__(
        self,
        chunk_size: int = 600,
        overlap: int = 100,
        extractor: str = "rules",
        max_llm_units: int | None = None,
        resume: bool = True,
    ) -> None:
        """Configure chunking, extraction backend, and LLM resume behavior."""
        if extractor not in {"rules", "llm"}:
            raise ValueError("extractor must be 'rules' or 'llm'")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.extractor = extractor
        self.max_llm_units = max_llm_units
        self.resume = resume

    def index(self, input_path: Path, output: Path) -> dict[str, int]:
        """Index an input file or directory and export all graph tables."""
        output.mkdir(parents=True, exist_ok=True)
        _load_dotenv(Path(".env"))
        documents = ingest(input_path)
        units = chunk_documents(documents, self.chunk_size, self.overlap)
        extraction_stats: dict[str, int] = {}
        if self.extractor == "llm":
            config = OpenRouterConfig.from_env()
            config.cache_dir = output / "llm_cache"
            llm_client = OpenRouterClient(config)
        else:
            llm_client = None
        entities, edges = extract_graph(
            documents,
            units,
            extractor=self.extractor,
            llm_client=llm_client,
            max_llm_units=self.max_llm_units,
            llm_cache_dir=output / "llm_cache" if self.extractor == "llm" else None,
            resume=self.resume,
            continue_on_error=llm_client.config.continue_on_error if llm_client else True,
            extraction_stats=extraction_stats,
        )
        calibrate_edges(edges)
        communities, reports = build_communities(entities, edges)

        tables = {
            "documents": [asdict(item) for item in documents],
            "text_units": [asdict(item) for item in units],
            "entities": [asdict(item) for item in entities],
            "relationships": [asdict(item) for item in edges],
            "calibrated_edges": [asdict(item) for item in edges],
            "communities": [asdict(item) for item in communities],
            "community_reports": [asdict(item) for item in reports],
        }
        for name, rows in tables.items():
            export_table(output, name, rows)
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "extractor": self.extractor,
            "llm_model": os.getenv("OPENROUTER_MODEL") if self.extractor == "llm" else None,
            "max_llm_units": self.max_llm_units,
            "extracted_text_units": min(len(units), self.max_llm_units) if self.max_llm_units else len(units),
            "resume": self.resume,
            **extraction_stats,
            "input": str(input_path.resolve()),
            "input_hash": _input_hash(input_path),
            "counts": {name: len(rows) for name, rows in tables.items()},
        }
        write_json(output / "manifest.json", manifest)
        return {**manifest["counts"], **extraction_stats}


def _input_hash(input_path: Path) -> str:
    """Hash index input contents for manifest reproducibility."""
    digest = hashlib.sha256()
    paths = [input_path] if input_path.is_file() else sorted(input_path.rglob("*"))
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(input_path) if input_path.is_dir() else path.name).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_dotenv(path: Path) -> None:
    """Load environment variables from a dotenv file, overwriting old values."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value
