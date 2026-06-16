from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries as UTF-8 JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSON Lines and return an empty list for missing files."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    """Write a JSON value with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def export_table(output: Path, name: str, rows: list[dict[str, Any]]) -> None:
    """Export a table to JSONL and, when pandas is available, Parquet."""
    write_jsonl(output / f"{name}.jsonl", rows)
    try:
        import pandas as pd

        parquet_rows = [
            {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, dict)
                else value
                for key, value in row.items()
            }
            for row in rows
        ]
        pd.DataFrame(parquet_rows).to_parquet(output / f"{name}.parquet", index=False)
    except (ImportError, ValueError):
        return
