from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ecgraphrag.dataset import _download, create_qa_splits, normalize_multihop_rag, normalize_musique_ans
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

    def test_musique_normalization_keeps_supporting_evidence_covered(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "musique_ans_dev.jsonl"
            raw.write_text(
                """{"id":"q1","question":"Who wrote the work connected to Beta?","answer":"Alice","paragraphs":[{"idx":0,"title":"Alpha","paragraph_text":"Alice wrote the Alpha work that is connected to Beta through a documented multihop relation.","is_supporting":true},{"idx":1,"title":"Distractor","paragraph_text":"This distractor paragraph has enough text to normalize but should not become gold evidence.","is_supporting":false}],"question_decomposition":[{"question":"Who wrote Alpha?","answer":"Alice"}]}""",
                encoding="utf-8",
            )
            documents, qa = normalize_musique_ans(raw, limit_qa=1)
            document_ids = {row["id"] for row in documents}
            evidence_ids = {item["url"] for item in qa[0]["evidence"]}
            self.assertEqual(len(qa), 1)
            self.assertTrue(evidence_ids)
            self.assertTrue(evidence_ids <= document_ids)
            self.assertEqual(qa[0]["metadata"]["dataset"], "musique_ans")

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
