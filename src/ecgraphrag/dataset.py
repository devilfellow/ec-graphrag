from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request

from .storage import write_jsonl, write_json

MULTIHOP_REPO = "https://github.com/yixuantt/MultiHop-RAG"
GITHUB_API_DATASET = "https://api.github.com/repos/yixuantt/MultiHop-RAG/contents/dataset"
RAW_BASE = "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset"
MUSIQUE_REPO = "https://github.com/stonybrooknlp/musique"
MUSIQUE_ANS_DEV_FILE_ID = "1TRXU68wveSehVbQrRRtWsUsFkKUF43QS"

TEXT_KEYS = ["text", "content", "body", "article", "passage", "document", "news", "article_text"]
TITLE_KEYS = ["title", "headline", "name"]
QUERY_KEYS = ["query", "question"]
ANSWER_KEYS = ["answer", "gold_answer", "ground_truth", "ground_truth_answer"]
EVIDENCE_KEYS = ["evidence", "supporting_evidence", "support", "supporting_facts", "evidence_list"]


def download_multihop_rag_dataset(output_dir: Path, limit_docs: int | None = None, limit_qa: int | None = None) -> dict[str, Any]:
    """Download the MultiHop-RAG dataset directory from GitHub and normalize it.

    The upstream repository exposes a `dataset/` folder. This function uses the
    GitHub contents API, saves raw files in `raw/`, then writes normalized files:
    `documents.jsonl` and `qa.jsonl`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = _github_dataset_files()
    downloaded: list[Path] = []
    for item in files:
        url = item.get("download_url")
        name = item.get("name")
        if not url or not name:
            continue
        target = raw_dir / name
        _download(url, target)
        downloaded.append(target)

    docs, qa = normalize_multihop_rag(raw_dir, limit_docs=limit_docs, limit_qa=limit_qa)
    write_jsonl(output_dir / "documents.jsonl", docs)
    write_jsonl(output_dir / "qa.jsonl", qa)
    manifest = {
        "source_repo": MULTIHOP_REPO,
        "raw_files": [path.name for path in downloaded],
        "documents": len(docs),
        "qa": len(qa),
        "evidence_coverage": _evidence_coverage(docs, qa),
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    return manifest


def download_musique_ans_dataset(output_dir: Path, limit_qa: int | None = None) -> dict[str, Any]:
    """Download MuSiQue-Ans dev and normalize it into the project QA format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "musique_ans_dev.jsonl"
    if not raw_path.exists():
        _download_google_drive_file(MUSIQUE_ANS_DEV_FILE_ID, raw_path)
    docs, qa = normalize_musique_ans(raw_path, limit_qa=limit_qa)
    write_jsonl(output_dir / "documents.jsonl", docs)
    write_jsonl(output_dir / "qa.jsonl", qa)
    manifest = {
        "source_repo": MUSIQUE_REPO,
        "source_file": "raw_data/musique_ans_dev.jsonl",
        "documents": len(docs),
        "qa": len(qa),
        "evidence_coverage": _evidence_coverage(docs, qa),
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    return manifest


def normalize_musique_ans(raw_path: Path, limit_qa: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize MuSiQue-Ans rows into document and QA rows."""
    rows = _read_rows(raw_path)
    selected_rows = rows[:limit_qa] if limit_qa else rows
    documents_by_id: dict[str, dict[str, Any]] = {}
    qa: list[dict[str, Any]] = []
    for row_index, row in enumerate(selected_rows):
        question_id = str(row.get("id") or f"musique:{row_index}")
        evidence: list[dict[str, Any]] = []
        for paragraph in _musique_paragraphs(row):
            if not isinstance(paragraph, dict):
                continue
            text = str(paragraph.get("paragraph_text") or paragraph.get("text") or "").strip()
            if len(text) < 30:
                continue
            title = str(paragraph.get("title") or paragraph.get("wikipedia_title") or f"paragraph-{paragraph.get('idx', len(documents_by_id))}")
            paragraph_idx = paragraph.get("idx", paragraph.get("wikipedia_id", len(documents_by_id)))
            doc_id = _musique_document_id(title, text, paragraph_idx)
            documents_by_id.setdefault(doc_id, {
                "id": doc_id,
                "title": title,
                "text": text,
                "source": "musique_ans_dev.jsonl",
                "metadata": {
                    "id": doc_id,
                    "title": title,
                    "source": "musique_ans_dev",
                    "question_id": question_id,
                    "paragraph_idx": paragraph_idx,
                    "is_supporting": bool(paragraph.get("is_supporting")),
                },
            })
            if paragraph.get("is_supporting"):
                evidence.append({
                    "url": doc_id,
                    "title": title,
                    "paragraph_idx": paragraph_idx,
                })
        query = _musique_question(row)
        answer = _musique_answer(row)
        if not query or answer is None:
            continue
        qa.append({
            "id": question_id,
            "query": str(query),
            "answer": str(answer),
            "evidence": evidence,
            "source": "musique_ans_dev.jsonl",
            "metadata": {
                "dataset": "musique_ans",
                "answerable": row.get("answerable", True),
                "hop_count": len(evidence),
                "question_decomposition": _musique_decomposition(row),
                "source_id": question_id,
            },
        })
    return list(documents_by_id.values()), qa


def normalize_multihop_rag(raw_dir: Path, limit_docs: int | None = None, limit_qa: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize raw MultiHop-RAG files into document and QA rows."""
    documents: list[dict[str, Any]] = []
    qa: list[dict[str, Any]] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".tsv"}:
            continue
        rows = _read_rows(path)
        kind = _guess_kind(path.name, rows)
        if kind == "documents":
            for row in rows:
                doc = _normalize_document(row, path.name)
                if doc:
                    documents.append(doc)
        elif kind == "qa":
            for row in rows:
                item = _normalize_qa(row, path.name)
                if item:
                    qa.append(item)
    documents = _dedupe_documents(documents)
    selected_qa = qa[:limit_qa] if limit_qa else qa
    selected_documents = _select_evidence_aligned_documents(documents, selected_qa, limit_docs)
    return selected_documents, selected_qa


def _select_evidence_aligned_documents(
    documents: list[dict[str, Any]],
    qa: list[dict[str, Any]],
    limit_docs: int | None,
) -> list[dict[str, Any]]:
    """Prefer evidence documents before filling the document limit."""
    if limit_docs is None:
        return documents
    evidence_urls = {
        str(item.get("url"))
        for row in qa
        for item in row.get("evidence", [])
        if isinstance(item, dict) and item.get("url")
    }
    evidence_titles = {
        str(item.get("title")).casefold()
        for row in qa
        for item in row.get("evidence", [])
        if isinstance(item, dict) and item.get("title")
    }
    preferred = [
        row for row in documents
        if row["id"] in evidence_urls
        or str(row.get("title", "")).casefold() in evidence_titles
    ]
    selected_ids = {row["id"] for row in preferred}
    remaining = [row for row in documents if row["id"] not in selected_ids]
    return preferred + remaining[:max(0, limit_docs - len(preferred))]


def _evidence_coverage(
    documents: list[dict[str, Any]],
    qa: list[dict[str, Any]],
) -> dict[str, int]:
    """Count QA rows with full or partial evidence coverage."""
    document_ids = {row["id"] for row in documents}
    document_titles = {str(row.get("title", "")).casefold() for row in documents}
    complete = 0
    partial = 0
    for row in qa:
        evidence = [item for item in row.get("evidence", []) if isinstance(item, dict)]
        if not evidence:
            continue
        matches = sum(
            str(item.get("url") or "") in document_ids
            or str(item.get("title") or "").casefold() in document_titles
            for item in evidence
        )
        complete += matches == len(evidence)
        partial += matches > 0
    return {"complete_qa": complete, "partial_qa": partial, "total_qa": len(qa)}


def _github_dataset_files() -> list[dict[str, Any]]:
    request = Request(GITHUB_API_DATASET, headers={"User-Agent": "ec-graphrag"})
    with urlopen(request, timeout=60) as response:  # nosec - public dataset API
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("Unexpected GitHub API response for MultiHop-RAG dataset")
    return [item for item in data if item.get("type") == "file"]


def _download(url: str, target: Path) -> None:
    data = _read_url(url)
    if data.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        media_url = url.replace(
            "https://raw.githubusercontent.com/",
            "https://media.githubusercontent.com/media/",
            1,
        )
        if media_url == url:
            raise RuntimeError(f"Cannot resolve Git LFS URL: {url}")
        data = _read_url(media_url)
    target.write_bytes(data)


def _download_google_drive_file(file_id: str, target: Path) -> None:
    """Download a public Google Drive file without requiring gdown."""
    base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    request = Request(base_url, headers={"User-Agent": "ec-graphrag"})
    with urlopen(request, timeout=120) as response:  # nosec - public dataset URL
        data = response.read()
        cookies = response.headers.get_all("Set-Cookie", [])
    if not _looks_like_google_drive_interstitial(data):
        target.write_bytes(data)
        return
    token = _google_drive_confirm_token(data)
    if not token:
        raise RuntimeError("Google Drive confirmation token was not found")
    confirm_url = f"{base_url}&confirm={token}"
    headers = {"User-Agent": "ec-graphrag"}
    if cookies:
        headers["Cookie"] = "; ".join(cookie.split(";", 1)[0] for cookie in cookies)
    confirm_request = Request(confirm_url, headers=headers)
    with urlopen(confirm_request, timeout=300) as response:  # nosec - public dataset URL
        target.write_bytes(response.read())


def _looks_like_google_drive_interstitial(data: bytes) -> bool:
    return data[:512].lstrip().startswith(b"<!DOCTYPE html") or b"confirm=" in data[:8192]


def _google_drive_confirm_token(data: bytes) -> str | None:
    text = data.decode("utf-8", errors="ignore")
    match = re.search(r"confirm=([0-9A-Za-z_]+)", text)
    return match.group(1) if match else None


def _read_url(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "ec-graphrag"})
    with urlopen(request, timeout=120) as response:  # nosec - public dataset URL
        return response.read()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read JSON, JSONL, CSV, or TSV rows from a raw dataset file."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(value, dict):
            for key in ("data", "documents", "queries", "qa", "items"):
                if isinstance(value.get(key), list):
                    return value[key]
            return [value]
        return value if isinstance(value, list) else []
    delimiter = "\t" if suffix == ".tsv" else ","
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def _guess_kind(filename: str, rows: list[dict[str, Any]]) -> str:
    """Classify a raw file as documents or QA rows."""
    lowered = filename.lower()
    if any(token in lowered for token in ("query", "question", "qa", "answer")):
        return "qa"
    if any(token in lowered for token in ("corpus", "document", "article", "news")):
        return "documents"
    if not rows:
        return "documents"
    keys = {str(key).lower() for row in rows[:5] for key in row.keys()}
    if keys & set(QUERY_KEYS) and keys & set(ANSWER_KEYS):
        return "qa"
    return "documents"


def _normalize_document(row: dict[str, Any], source_file: str) -> dict[str, Any] | None:
    """Normalize one raw document row."""
    text = _first(row, TEXT_KEYS)
    if not text:
        values = [str(value) for value in row.values() if value is not None]
        text = " ".join(values)
    text = str(text).strip()
    if len(text) < 30:
        return None
    doc_id = str(row.get("id") or row.get("doc_id") or row.get("document_id") or row.get("url") or f"{source_file}:{abs(hash(text[:256]))}")
    return {
        "id": doc_id,
        "title": str(_first(row, TITLE_KEYS) or doc_id),
        "text": text,
        "source": source_file,
        "metadata": {k: v for k, v in row.items() if k not in TEXT_KEYS},
    }


def _normalize_qa(row: dict[str, Any], source_file: str) -> dict[str, Any] | None:
    """Normalize one raw question-answer row."""
    query = _first(row, QUERY_KEYS)
    answer = _first(row, ANSWER_KEYS)
    if not query or answer is None:
        return None
    evidence = _first(row, EVIDENCE_KEYS) or []
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except json.JSONDecodeError:
            evidence = [evidence]
    return {
        "id": str(row.get("id") or row.get("query_id") or f"{source_file}:{abs(hash(str(query)))}"),
        "query": str(query),
        "answer": str(answer),
        "evidence": evidence,
        "source": source_file,
        "metadata": row,
    }


def _first(row: dict[str, Any], keys: list[str]) -> Any:
    lower_to_key = {str(key).lower(): key for key in row.keys()}
    for key in keys:
        if key in lower_to_key:
            value = row[lower_to_key[key]]
            if value not in (None, ""):
                return value
    return None


def _dedupe_documents(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = row["id"]
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _musique_document_id(title: str, text: str, paragraph_idx: Any) -> str:
    import hashlib

    value = f"{title}\n{paragraph_idx}\n{text}".encode("utf-8")
    return "musique:" + hashlib.sha1(value).hexdigest()[:16]


def _musique_paragraphs(row: dict[str, Any]) -> list[dict[str, Any]]:
    paragraphs = row.get("paragraphs")
    if isinstance(paragraphs, list):
        return [item for item in paragraphs if isinstance(item, dict)]
    contexts = row.get("contexts")
    if isinstance(contexts, list):
        return [item for item in contexts if isinstance(item, dict)]
    return []


def _musique_question(row: dict[str, Any]) -> Any:
    return row.get("question") or row.get("composed_question_text") or row.get("question_text") or row.get("query_text")


def _musique_answer(row: dict[str, Any]) -> Any:
    return row.get("answer") if row.get("answer") is not None else row.get("answer_text")


def _musique_decomposition(row: dict[str, Any]) -> list[Any]:
    if isinstance(row.get("question_decomposition"), list):
        return row["question_decomposition"]
    if isinstance(row.get("decomposed_instances"), list):
        return [
            {
                "question": item.get("question_text"),
                "answer": item.get("answer_text"),
            }
            for item in row["decomposed_instances"]
            if isinstance(item, dict)
        ]
    return []


def create_qa_splits(
    qa_path: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
) -> dict[str, int]:
    """Create reproducible train/dev/test QA splits stratified by question type."""
    rows = _read_rows(qa_path)
    groups: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        question_type = str(metadata.get("question_type") or "unknown")
        evidence_count = len(row.get("evidence") or [])
        groups[(question_type, evidence_count)].append(row)
    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for key in sorted(groups):
        values = list(groups[key])
        rng.shuffle(values)
        train_end = round(len(values) * train_ratio)
        dev_end = train_end + round(len(values) * dev_ratio)
        splits["train"].extend(values[:train_end])
        splits["dev"].extend(values[train_end:dev_end])
        splits["test"].extend(values[dev_end:])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        rng.shuffle(values)
        write_jsonl(output_dir / f"{name}.jsonl", values)
    manifest = {
        "seed": seed,
        "train_ratio": train_ratio,
        "dev_ratio": dev_ratio,
        "test_ratio": round(1.0 - train_ratio - dev_ratio, 6),
        **{name: len(values) for name, values in splits.items()},
    }
    write_json(output_dir / "manifest.json", manifest)
    return {name: len(values) for name, values in splits.items()}


def main() -> None:
    """Run dataset normalization and optional QA split creation from the CLI."""
    parser = argparse.ArgumentParser(prog="ecgraphrag.dataset")
    parser.add_argument("--name", choices=["multihop_rag", "musique_ans"], default="multihop_rag")
    parser.add_argument("--output", type=Path, default=Path("data/multihop_rag"))
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--limit-qa", type=int)
    parser.add_argument("--split-output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.split_output:
        print(json.dumps(create_qa_splits(args.output / "qa.jsonl", args.split_output, args.seed), indent=2))
        return
    if args.name == "musique_ans":
        manifest = download_musique_ans_dataset(args.output, args.limit_qa)
    else:
        manifest = download_multihop_rag_dataset(args.output, args.limit_docs, args.limit_qa)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
