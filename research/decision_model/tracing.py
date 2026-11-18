"""Jev operation records for the shared PaperQA experiment logger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from paperqa.decision_models.jev import build_relevance_request

from research.baseline.tracing import TraceWriter
from research.baseline.utils import redact_secrets


def _safe_error(error: Exception) -> str:
    return redact_secrets(f"{type(error).__name__}: {error}")


@dataclass
class JevTotals:
    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    known_api_attempts: int = 0

    @property
    def cost_complete(self) -> bool:
        return self.failed_calls == 0


class JevTraceCollector:
    def __init__(self, writer: TraceWriter, settings: Any) -> None:
        self.writer = writer
        self.settings = settings
        self.totals = JevTotals()
        self.step = 0

    def _record_success(self, *, operation: str, step: int, result: Any,
                        input_data: dict[str, Any], output_data: dict[str, Any],
                        decision_type: str, **extra: Any) -> None:
        self.totals.calls += 1
        self.totals.input_tokens += result.input_tokens
        self.totals.output_tokens += result.output_tokens
        self.totals.cost_usd += result.cost_usd
        self.totals.known_api_attempts += 1 + result.retry_count
        self.writer.emit(
            "jev_call", step=step, operation=operation, cognitive_mode="decision",
            backend="jev", model=result.model, decision_type=decision_type,
            input=input_data, output=output_data, latency_ms=result.latency_ms,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            cost_usd=result.cost_usd, confidence=result.confidence,
            probabilities=result.probabilities, retry_count=result.retry_count,
            api_attempts=1 + result.retry_count,
            error=None, timeout=False, **extra,
        )

    def _record_error(self, *, operation: str, step: int, model: str,
                      input_data: dict[str, Any], error: Exception,
                      latency_ms: float, decision_type: str, **extra: Any) -> None:
        self.totals.calls += 1
        self.totals.failed_calls += 1
        self.writer.emit(
            "jev_call_failed", step=step, operation=operation,
            cognitive_mode="decision", backend="jev", model=model,
            decision_type=decision_type, input=input_data, output=None,
            latency_ms=latency_ms, input_tokens=None, output_tokens=None,
            cost_usd=None, confidence=None, probabilities=None,
            retry_count=None, error=_safe_error(error),
            api_attempts=None,
            timeout=isinstance(error, (TimeoutError, httpx.TimeoutException)), **extra,
        )

    async def decision_result(self, result: Any, _state: Any) -> None:
        completion = result.choice in {"complete_success", "complete_unsure"}
        self._record_success(
            operation="termination" if completion else "tool_selection",
            step=self.step + 1, result=result,
            input_data=result.request or {},
            output_data={"choice": result.choice}, decision_type="choice",
            selected_tool="complete" if completion else result.choice,
        )

    async def decision_error(
        self, request: dict[str, Any], error: Exception, latency_ms: float, _state: Any
    ) -> None:
        self._record_error(
            operation="tool_selection", step=self.step + 1,
            model=self.settings.agent.jev_decision_model,
            input_data=request, error=error, latency_ms=latency_ms,
            decision_type="choice",
        )

    def relevance_result(self, context: Any, result: Any) -> None:
        question = context.question or ""
        evidence = context.text.text.strip("\n") or "(no text)"
        source = {
            "context_id": context.id,
            "chunk_name": context.text.name,
            "docname": context.text.doc.docname,
            "dockey": context.text.doc.dockey,
        }
        self._record_success(
            operation="evidence_relevance", step=self.step, result=result,
            input_data={**build_relevance_request(question, evidence, result.model), **source},
            output_data={"score": result.score, "raw_score": result.raw_score,
                         "summary": context.context},
            decision_type="score", score=result.score, raw_score=result.raw_score,
            **source,
        )

    def relevance_error(
        self, text: Any, question: str, evidence: str,
        error: Exception, latency_ms: float,
    ) -> None:
        self._record_error(
            operation="evidence_relevance", step=self.step,
            model=self.settings.answer.jev_relevance_model,
            input_data={
                **build_relevance_request(
                    question, evidence, self.settings.answer.jev_relevance_model
                ),
                "chunk_name": text.name,
                "docname": text.doc.docname,
                "dockey": text.doc.dockey,
            },
            error=error, latency_ms=latency_ms, decision_type="score",
            chunk_name=text.name,
        )
