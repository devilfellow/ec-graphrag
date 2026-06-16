from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ecgraphrag.dataset import _download, create_qa_splits, normalize_multihop_rag
from ecgraphrag.storage import read_jsonl, write_jsonl


class DatasetDownloadTest(unittest.TestCase):
    def test_download_resolves_github_lfs_pointer(self) -> None:
        raw_url = "https://raw.githubusercontent.com/example/repo/main/dataset/data.json"
        media_url = "https://media.githubusercontent.com/media/example/repo/main/dataset/data.json"
        pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 2\n"

        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "data.json"
            with patch(
                "ecgraphrag.dataset._read_url",
                side_effect=lambda url: pointer if url == raw_url else b"[]",
            ) as read_url:
                _download(raw_url, target)

            self.assertEqual(target.read_bytes(), b"[]")
            self.assertEqual([call.args[0] for call in read_url.call_args_list], [raw_url, media_url])

    def test_normalization_selects_qa_evidence_documents_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)
            (raw / "corpus.json").write_text(
                """[
                  {"url":"unrelated","title":"Unrelated","body":"An unrelated document with enough text to normalize."},
                  {"url":"gold","title":"Gold","body":"The gold evidence document with enough text to normalize."}
                ]""",
                encoding="utf-8",
            )
            (raw / "MultiHopRAG.json").write_text(
                """[{"query":"What is gold?","answer":"Gold","evidence_list":[{"url":"gold","title":"Gold","fact":"Gold fact"}]}]""",
                encoding="utf-8",
            )
            documents, qa = normalize_multihop_rag(raw, limit_docs=1, limit_qa=1)
            self.assertEqual(documents[0]["id"], "gold")
            self.assertEqual(qa[0]["answer"], "Gold")

    def test_qa_splits_are_reproducible_and_stratified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            qa = root / "qa.jsonl"
            rows = [
                {
                    "id": f"q{index}",
                    "query": f"Question {index}",
                    "answer": "Answer",
                    "evidence": [{"url": str(item)} for item in range(index % 3)],
                    "metadata": {"question_type": "inference" if index % 2 else "comparison"},
                }
                for index in range(40)
            ]
            write_jsonl(qa, rows)
            first = root / "first"
            second = root / "second"
            counts = create_qa_splits(qa, first, seed=42)
            create_qa_splits(qa, second, seed=42)
            self.assertEqual(sum(counts.values()), len(rows))
            for name in ("train", "dev", "test"):
                self.assertEqual(read_jsonl(first / f"{name}.jsonl"), read_jsonl(second / f"{name}.jsonl"))


if __name__ == "__main__":
    unittest.main()
