from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class OpenRouterConfig:
    """Runtime configuration for OpenRouter chat-completion calls."""

    model: str = "openai/gpt-4o-mini"
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 10000
    timeout: int = 90
    retries: int = 3
    site_url: str | None = None
    app_name: str = "EC-GraphRAG"
    repair_json: bool = True
    retry_with_shorter_input: bool = True
    cache_dir: Path | None = None
    continue_on_error: bool = True
    workers: int = 12

    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        """Create configuration from OpenRouter environment variables."""
        return cls(
            model=os.getenv("OPENROUTER_MODEL", cls.model),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=float(os.getenv("OPENROUTER_TEMPERATURE", "0")),
            max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", "10000")),
            timeout=int(os.getenv("OPENROUTER_TIMEOUT", "90")),
            retries=int(os.getenv("OPENROUTER_RETRIES", "3")),
            site_url=os.getenv("OPENROUTER_SITE_URL"),
            app_name=os.getenv("OPENROUTER_APP_NAME", "EC-GraphRAG"),
            repair_json=os.getenv("OPENROUTER_REPAIR_JSON", "true").casefold() == "true",
            retry_with_shorter_input=os.getenv("OPENROUTER_RETRY_SHORTER_INPUT", "true").casefold() == "true",
            cache_dir=Path(value) if (value := os.getenv("OPENROUTER_CACHE_DIR")) else None,
            continue_on_error=os.getenv("OPENROUTER_CONTINUE_ON_ERROR", "true").casefold() == "true",
            workers=max(1, int(os.getenv("OPENROUTER_WORKERS", "12"))),
        )


class OpenRouterClient:
    """Small OpenRouter client using the OpenAI-compatible Chat Completions API.

    The API key is intentionally read from the environment. Do not hard-code
    secrets in notebooks, scripts, or archives.
    """

    def __init__(self, config: OpenRouterConfig | None = None) -> None:
        """Initialize the client and validate that an API key is available."""
        if requests is None:
            raise RuntimeError("OpenRouter support requires the 'requests' package")
        self.config = config or OpenRouterConfig.from_env()
        self._diagnostic_lock = threading.Lock()
        if not self.config.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Create .env or export the variable before LLM indexing."
            )

    def chat_json(self, system: str, user: str, schema_hint: str | None = None) -> dict[str, Any]:
        """Request a JSON object from OpenRouter with diagnostics and retries."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.site_url or "https://localhost/ec-graphrag",
            "X-Title": self.config.app_name,
        }
        last_error: Exception | None = None
        malformed_content = ""
        for attempt in range(self.config.retries):
            attempt_user = user
            if attempt == 1 and malformed_content:
                attempt_user = (
                    user
                    + "\n\nThe previous response below was invalid JSON. Return the complete corrected JSON object only.\n"
                    + malformed_content[:6000]
                )
            elif attempt == self.config.retries - 1 and self.config.retry_with_shorter_input:
                attempt_user = _shorten_user_prompt(user)
            payload = self._payload(system, attempt_user, schema_hint)
            try:
                response = requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                response_payload = response.json()
                choice = response_payload["choices"][0]
                content = str(choice["message"]["content"])
                finish_reason = choice.get("finish_reason")
                try:
                    if finish_reason == "length":
                        raise json.JSONDecodeError("OpenRouter response was truncated", content, len(content))
                    value = _parse_json_object(content, repair=self.config.repair_json)
                except json.JSONDecodeError as exc:
                    malformed_content = content
                    self._write_diagnostic(
                        attempt,
                        "truncated_json" if finish_reason == "length" else "json_decode",
                        exc,
                        finish_reason,
                        response_payload.get("usage"),
                        content,
                    )
                    raise
                self._write_diagnostic(
                    attempt, "success", None, finish_reason, response_payload.get("usage"), ""
                )
                return value
            except Exception as exc:  # pragma: no cover - requires network/API
                last_error = exc
                if not isinstance(exc, json.JSONDecodeError):
                    self._write_diagnostic(attempt, _error_kind(exc), exc, None, None, "")
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"OpenRouter request failed after retries: {last_error}")

    def _payload(self, system: str, user: str, schema_hint: str | None) -> dict[str, Any]:
        """Build the OpenAI-compatible chat completion payload."""
        system_content = system
        if schema_hint:
            system_content += "\n\nJSON schema hint:\n" + schema_hint
        return {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }

    def _write_diagnostic(
        self,
        attempt: int,
        status: str,
        error: Exception | None,
        finish_reason: Any,
        usage: Any,
        content: str,
    ) -> None:
        """Append request diagnostics to the configured cache directory."""
        if not self.config.cache_dir:
            return
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.config.model,
            "attempt": attempt + 1,
            "status": status,
            "finish_reason": finish_reason,
            "usage": usage,
            "error": str(error) if error else None,
            "content": content,
        }
        with self._diagnostic_lock:
            with (self.config.cache_dir / "openrouter_diagnostics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_json_object(content: str, repair: bool = True) -> dict[str, Any]:
    """Parse an LLM JSON object, optionally repairing common truncation."""
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    try:
        value = json.loads(content)
    except json.JSONDecodeError as original_error:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            try:
                value = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                if not repair:
                    raise original_error
                value = json.loads(_strict_repair_json(content[start:]))
        elif repair and start != -1:
            value = json.loads(_strict_repair_json(content[start:]))
        else:
            raise original_error
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def _strict_repair_json(content: str) -> str:
    """Repair only structural JSON truncation without inventing missing fields."""
    repaired = content.rstrip()
    while repaired.endswith((",", ":")):
        repaired = repaired[:-1].rstrip()
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in repaired:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or (char == "]" and stack[-1] != "[") or (char == "}" and stack[-1] != "{"):
                raise json.JSONDecodeError("Mismatched JSON delimiter", repaired, 0)
            stack.pop()
    if in_string:
        repaired += '"'
    repaired += "".join("]" if char == "[" else "}" for char in reversed(stack))
    return repaired


def _shorten_user_prompt(user: str) -> str:
    """Shorten the text portion of a prompt for the final retry."""
    marker = "\nText:\n"
    if marker not in user:
        return user[:max(1000, len(user) // 2)]
    prefix, text = user.split(marker, 1)
    return prefix + marker + text[:max(1000, len(text) // 2)]


def _error_kind(exc: Exception) -> str:
    """Classify request exceptions for diagnostics."""
    name = type(exc).__name__.casefold()
    if "timeout" in name:
        return "timeout"
    if "http" in name:
        return "http"
    return "request_error"
