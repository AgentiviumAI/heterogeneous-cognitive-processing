from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_RETRY_PATTERNS = (
    re.compile(r"Please try again in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE),
    re.compile(r"retry[_ -]?after[\"'\s:=]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
)


def parse_retry_after_seconds(exc: BaseException) -> float | None:
    """Best-effort extraction of provider retry-after without secret logging."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        for name in ("retry-after", "Retry-After"):
            try:
                value = headers.get(name)
            except Exception:
                value = None
            if value is not None:
                try:
                    return max(0.0, float(value))
                except (TypeError, ValueError):
                    pass

    message = str(exc)
    for pattern in _RETRY_PATTERNS:
        match = pattern.search(message)
        if match:
            return max(0.0, float(match.group(1)))
    return None


def install_adaptive_rate_limit_recovery(
    *,
    max_attempts: int,
    default_wait_seconds: float,
    max_wait_seconds: float,
) -> Callable[[], None]:
    """Wrap ``litellm.Router.acompletion`` with final-429 wait/retry recovery.

    LiteLLM already rotates deployments and performs its own retries. This wrapper
    only acts after that machinery has exhausted the currently available pool.
    It waits for the provider's Retry-After when available, then asks the Router
    to select a healthy deployment again. The patch is process-local and the
    returned callable restores the original method.
    """
    import litellm

    if max_attempts <= 0:
        return lambda: None

    current = litellm.Router.acompletion
    if getattr(current, "__paperqa_adaptive_rate_limit__", False):
        return lambda: None

    original = current

    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        recovery_attempt = 0
        while True:
            try:
                return await original(self, *args, **kwargs)
            except litellm.RateLimitError as exc:
                recovery_attempt += 1
                if recovery_attempt > max_attempts:
                    raise

                provider_wait = parse_retry_after_seconds(exc)
                if provider_wait is None:
                    # Deterministic exponential fallback keeps research runs
                    # reproducible while still allowing a rolling window to heal.
                    provider_wait = default_wait_seconds * (2 ** (recovery_attempt - 1))
                wait_seconds = min(max_wait_seconds, max(0.1, provider_wait))

                logger.warning(
                    "Provider pool exhausted by rate limit; waiting %.3fs before "
                    "re-entering LiteLLM Router (recovery %d/%d).",
                    wait_seconds,
                    recovery_attempt,
                    max_attempts,
                )
                await asyncio.sleep(wait_seconds)

    setattr(wrapped, "__paperqa_adaptive_rate_limit__", True)
    setattr(wrapped, "__paperqa_original_acompletion__", original)
    litellm.Router.acompletion = wrapped

    def restore() -> None:
        if litellm.Router.acompletion is wrapped:
            litellm.Router.acompletion = original

    return restore
