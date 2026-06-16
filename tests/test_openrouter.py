from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ecgraphrag.openrouter import (
    OpenRouterClient,
    OpenRouterConfig,
    _parse_json_object,
)


class _Response:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.content = content
        self.finish_reason = finish_reason

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [{
                "message": {"content": self.content},
                "finish_reason": self.finish_reason,
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }


class OpenRouterTest(unittest.TestCase):
    def test_parse_markdown_json(self) -> None:
        self.assertEqual(_parse_json_object('```json\n{"entities":[],"relationships":[]}\n```'), {
            "entities": [],
            "relationships": [],
        })

    def test_strict_repair_closes_truncated_arrays(self) -> None:
        value = _parse_json_object('{"entities":[],"relationships":[{"source":"A","target":"B"}')
        self.assertEqual(value["relationships"][0]["target"], "B")

    def test_strict_repair_does_not_invent_missing_comma(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            _parse_json_object('{"entities":[] "relationships":[]}')

    def test_retry_changes_prompt_and_writes_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = OpenRouterConfig(
                api_key="test", retries=3, cache_dir=Path(temp), retry_with_shorter_input=True
            )
            client = OpenRouterClient(config)
            responses = [
                _Response('{"entities":[] "relationships":[]}'),
                _Response('{"entities":[] "relationships":[]}'),
                _Response('{"entities":[],"relationships":[]}'),
            ]
            with patch("ecgraphrag.openrouter.requests.post", side_effect=responses) as post:
                value = client.chat_json("system", "Header\nText:\n" + "x" * 4000)
            self.assertEqual(value["relationships"], [])
            prompts = [call.kwargs["json"]["messages"][1]["content"] for call in post.call_args_list]
            self.assertIn("previous response below was invalid JSON", prompts[1])
            self.assertLess(len(prompts[2]), len(prompts[0]))
            diagnostics = [
                json.loads(line)
                for line in (Path(temp) / "openrouter_diagnostics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["status"] for row in diagnostics], ["json_decode", "json_decode", "success"])

    def test_truncated_valid_json_is_retried(self) -> None:
        config = OpenRouterConfig(api_key="test", retries=2)
        client = OpenRouterClient(config)
        responses = [
            _Response('{"entities":[],"relationships":[]}', finish_reason="length"),
            _Response('{"entities":[],"relationships":[]}'),
        ]
        with patch("ecgraphrag.openrouter.requests.post", side_effect=responses) as post:
            value = client.chat_json("system", "user")
        self.assertEqual(value["entities"], [])
        self.assertEqual(post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
