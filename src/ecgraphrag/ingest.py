from __future__ import annotations

import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .models import Document, TextUnit
from .text import estimate_tokens, stable_id


class _HTMLTextExtractor(HTMLParser):
    """Collect visible text from simple HTML documents."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        """Collect non-empty text nodes."""
        if data.strip():
            self.parts.append(data.strip())


def _load_file(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Load supported file formats into text and metadata rows."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return [(path.read_text(encoding="utf-8", errors="replace"), {})]
    if suffix in {".html", ".htm"}:
        parser = _HTMLTextExtractor()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        return [(" ".join(parser.parts), {})]
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        return [(" ".join(str(value) for value in row.values()), row) for row in rows]
    if suffix in {".json", ".jsonl"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
        values = json.loads(raw) if suffix == ".json" else [
            json.loads(line) for line in raw.splitlines() if line.strip()
        ]
        values = values if isinstance(values, list) else [values]
        result = []
        for value in values:
            if isinstance(value, str):
                result.append((value, {}))
            else:
                text = str(value.get("text") or value.get("content") or value.get("body") or "")
                result.append((text, value))
        return result
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF ingest requires the optional 'pypdf' package") from exc
        reader = PdfReader(str(path))
        return [("\n".join(page.extract_text() or "" for page in reader.pages), {})]
    return []


def ingest(input_path: Path) -> list[Document]:
    """Load one file or a directory tree into normalized documents."""
    paths = [input_path] if input_path.is_file() else sorted(
        path for path in input_path.rglob("*") if path.is_file()
    )
    documents: list[Document] = []
    for path in paths:
        for index, (text, metadata) in enumerate(_load_file(path)):
            clean = re.sub(r"\s+", " ", text).strip()
            if not clean:
                continue
            doc_id = stable_id("doc", str(path.resolve()), str(index), clean[:256])
            title = str(metadata.get("title") or path.stem)
            documents.append(Document(doc_id, title, clean, str(path), metadata))
    return documents


def chunk_documents(
    documents: list[Document], chunk_size: int = 600, overlap: int = 100
) -> list[TextUnit]:
    """Split documents into overlapping word chunks."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be in [0, chunk_size)")
    units: list[TextUnit] = []
    stride = chunk_size - overlap
    for document in documents:
        words = re.findall(r"\S+", document.text, re.UNICODE)
        for position, start in enumerate(range(0, len(words), stride)):
            part = words[start : start + chunk_size]
            if not part:
                break
            text = " ".join(part)
            unit_id = stable_id("tu", document.id, str(position), text)
            units.append(TextUnit(unit_id, document.id, text, position, estimate_tokens(text)))
            if start + chunk_size >= len(words):
                break
    return units
