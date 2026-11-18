from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_secrets(value: str) -> str:
    for name in ("GROQ_API_KEY", "GEMINI_API_KEY", "TYPESAFE_API_KEY", "OPENAI_API_KEY"):
        secret = os.getenv(name)
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value


class RedactedStream:
    """Redact configured API keys from provider diagnostics on stderr."""

    def __init__(self, stream: Any) -> None:
        self.stream = stream

    def write(self, value: str) -> int:
        self.stream.write(redact_secrets(value))
        return len(value)

    def flush(self) -> None:
        self.stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


def dump_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", exclude_none=True)
        except TypeError:
            return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {str(k): dump_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [dump_jsonable(v) for v in value]
    return str(value)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dump_jsonable(record), ensure_ascii=False) + "\n")
        handle.flush()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def settings_fingerprint(settings: Any) -> str:
    payload = settings.model_dump_json(
        exclude={"agent": {"callbacks"}},
        exclude_none=False,
    ).encode()
    return hashlib.md5(payload).hexdigest()  # noqa: S324 - reproducibility fingerprint


def stable_user_question_id(question: str) -> str:
    return "user-" + hashlib.sha256(question.encode()).hexdigest()[:12]
