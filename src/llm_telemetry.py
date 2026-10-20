"""Structured, append-only telemetry for LLM calls and pipeline spans."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


TELEMETRY_EVENT_VERSION = "llm_telemetry_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _first_int(value: Any, names: Iterable[str]) -> Optional[int]:
    for name in names:
        candidate = _read(value, name)
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def estimate_tokens(value: Any) -> int:
    """Conservative tokenizer-free fallback used only when provider usage is absent."""
    if value is None:
        return 0
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            value = str(value)
    # Four UTF-8-ish characters per token is a common English approximation.
    return max(1, int(math.ceil(len(value) / 4.0))) if value else 0


def extract_response_metadata(response: Any, provider: str) -> Dict[str, Any]:
    """Normalize OpenAI/Azure and Gemini response usage without importing either SDK."""
    metadata: Dict[str, Any] = {
        "resolved_model": _read(response, "model") or _read(response, "model_version"),
        "response_id": _read(response, "id") or _read(response, "response_id"),
        "system_fingerprint": _read(response, "system_fingerprint"),
        "service_tier": _read(response, "service_tier"),
    }

    # Groq exposes an OpenAI-compatible Chat Completions response, including
    # ``usage.prompt_tokens``, ``usage.completion_tokens``, and token-detail
    # objects. Treat it like the OpenAI providers so telemetry records the
    # provider's billable counts instead of falling back to character-based
    # estimates.
    if provider in {"openai", "azure_openai", "groq"}:
        choices = _read(response, "choices") or []
        if choices:
            metadata["finish_reason"] = _read(choices[0], "finish_reason")
        usage = _read(response, "usage")
        if usage is not None:
            metadata.update(
                {
                    "input_tokens": _first_int(usage, ("prompt_tokens", "input_tokens")),
                    "output_tokens": _first_int(usage, ("completion_tokens", "output_tokens")),
                    "total_tokens": _first_int(usage, ("total_tokens",)),
                }
            )
            prompt_details = _read(usage, "prompt_tokens_details") or _read(usage, "input_tokens_details")
            completion_details = _read(usage, "completion_tokens_details") or _read(usage, "output_tokens_details")
            metadata["cached_input_tokens"] = _first_int(
                prompt_details,
                ("cached_tokens", "cached_input_tokens"),
            )
            metadata["reasoning_output_tokens"] = _first_int(
                completion_details,
                ("reasoning_tokens", "reasoning_output_tokens"),
            )
            metadata["usage_source"] = "provider"
    elif provider == "google_genai":
        candidates = _read(response, "candidates") or []
        if candidates:
            metadata["finish_reason"] = str(_read(candidates[0], "finish_reason") or "") or None
        usage = _read(response, "usage_metadata") or _read(response, "usage")
        if usage is not None:
            metadata.update(
                {
                    "input_tokens": _first_int(
                        usage,
                        ("prompt_token_count", "prompt_tokens", "input_tokens"),
                    ),
                    "output_tokens": _first_int(
                        usage,
                        ("candidates_token_count", "completion_tokens", "output_tokens"),
                    ),
                    "total_tokens": _first_int(
                        usage,
                        ("total_token_count", "total_tokens"),
                    ),
                    "cached_input_tokens": _first_int(
                        usage,
                        ("cached_content_token_count", "cached_input_tokens"),
                    ),
                    "reasoning_output_tokens": _first_int(
                        usage,
                        ("thoughts_token_count", "reasoning_tokens"),
                    ),
                    "usage_source": "provider",
                }
            )

    return {key: value for key, value in metadata.items() if value is not None}


def request_fingerprint(messages: Any, response_schema: Any = None) -> Dict[str, Any]:
    """Return non-content request diagnostics suitable for reproducibility logs."""
    try:
        messages_json = json.dumps(messages, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        messages_json = str(messages)
    try:
        schema_json = (
            json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"), default=str)
            if response_schema is not None
            else ""
        )
    except Exception:
        schema_json = str(response_schema or "")
    combined = f"{messages_json}\n{schema_json}"
    return {
        "message_count": len(messages) if isinstance(messages, list) else None,
        "prompt_characters": len(messages_json),
        "schema_characters": len(schema_json),
        "estimated_input_tokens": estimate_tokens(combined),
        "request_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
    }


def make_telemetry_path(
    root: Path,
    *,
    stage: str,
    participant_id: Optional[str] = None,
) -> Path:
    """Create a collision-resistant per-process JSONL path beneath ``root``."""
    safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage).strip("_") or "unknown"
    safe_participant = re.sub(r"[^A-Za-z0-9_.-]+", "_", participant_id or "all").strip("_") or "all"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(root) / "llm_telemetry" / f"{safe_stage}_{safe_participant}_{stamp}_{os.getpid()}.jsonl"


class LLMTelemetryRecorder:
    """Small synchronized JSONL recorder with mutable per-client context."""

    def __init__(
        self,
        path: Path,
        *,
        base_context: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or str(uuid.uuid4())
        self.base_context = dict(base_context or {})
        self.context: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def set_context(self, **context: Any) -> None:
        self.context.update({key: value for key, value in context.items() if value is not None})

    def clear_context(self, *keys: str) -> None:
        if keys:
            for key in keys:
                self.context.pop(key, None)
        else:
            self.context.clear()

    def record(self, event: Dict[str, Any]) -> None:
        payload = {
            "event_version": TELEMETRY_EVENT_VERSION,
            "recorded_at": _utc_now(),
            "run_id": self.run_id,
            **self.base_context,
            **self.context,
            **event,
        }
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
