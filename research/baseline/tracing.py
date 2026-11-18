from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from .paperqa_settings import ModelBundle
from .utils import append_jsonl, dump_jsonable, redact_secrets, utc_now


class TraceWriter:
    def __init__(self, path: Path, run_id: str, question_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.question_id = question_id
        self.started = time.perf_counter()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def emit(self, event: str, **payload: Any) -> None:
        append_jsonl(
            self.path,
            {
                "run_id": self.run_id,
                "question_id": self.question_id,
                "timestamp": utc_now(),
                "elapsed_ms": round((time.perf_counter() - self.started) * 1000, 3),
                "event": event,
                **payload,
            },
        )


@dataclass
class CallTotals:
    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_observations: int = 0


class BaselineLLMLogger(CustomLogger):
    """Provider-independent per-call trace using LiteLLM's callback surface."""

    def __init__(self) -> None:
        super().__init__()
        self.writer: TraceWriter | None = None
        self.totals = CallTotals()
        self.operation = "agent_step"
        self.cognitive_mode = "mixed"
        self.model_metadata: dict[str, dict[str, Any]] = {}

    def begin_question(self, writer: TraceWriter, bundle: ModelBundle) -> None:
        self.writer = writer
        self.totals = CallTotals()
        self.operation = "agent_step"
        self.cognitive_mode = "mixed"
        self.model_metadata = {
            bundle.answer.litellm_model: bundle.answer.public_metadata(),
            bundle.summary.litellm_model: bundle.summary.public_metadata(),
            bundle.agent.litellm_model: bundle.agent.public_metadata(),
        }

    def set_operation(self, operation: str, cognitive_mode: str) -> None:
        self.operation = operation
        self.cognitive_mode = cognitive_mode

    @staticmethod
    def _duration_ms(start_time: Any, end_time: Any) -> float | None:
        try:
            return round((end_time - start_time).total_seconds() * 1000, 3)
        except Exception:
            return None

    @staticmethod
    def _usage(response_obj: Any) -> tuple[int, int]:
        usage = getattr(response_obj, "usage", None)
        if usage is None and isinstance(response_obj, dict):
            usage = response_obj.get("usage")
        if usage is None:
            return 0, 0

        def get(name: str) -> int:
            if isinstance(usage, dict):
                return int(usage.get(name) or 0)
            return int(getattr(usage, name, 0) or 0)

        return get("prompt_tokens"), get("completion_tokens")

    @staticmethod
    def _cost(kwargs: dict[str, Any], response_obj: Any) -> float | None:
        hidden = getattr(response_obj, "_hidden_params", {}) if response_obj else {}
        for value in (
            kwargs.get("response_cost"),
            (kwargs.get("litellm_params") or {}).get("response_cost"),
            hidden.get("response_cost") if isinstance(hidden, dict) else None,
        ):
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _provider_metadata(self, model: str | None) -> dict[str, Any] | None:
        if model and model in self.model_metadata:
            return self.model_metadata[model]
        if len(self.model_metadata) == 1:
            return next(iter(self.model_metadata.values()))
        return None

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        if self.writer is None:
            return
        input_tokens, output_tokens = self._usage(response_obj)
        cost = self._cost(kwargs, response_obj)
        model = getattr(response_obj, "model", None) or kwargs.get("model")
        self.totals.calls += 1
        self.totals.input_tokens += input_tokens
        self.totals.output_tokens += output_tokens
        if cost is not None:
            self.totals.cost_usd += cost
            self.totals.cost_observations += 1

        self.writer.emit(
            "llm_call",
            operation=self.operation,
            cognitive_mode=self.cognitive_mode,
            backend="llm",
            model=model,
            provider=self._provider_metadata(model),
            latency_ms=self._duration_ms(start_time, end_time),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            input={"messages": kwargs.get("messages")},
            output=dump_jsonable(response_obj),
            retry_count=None,
            error=None,
        )

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        if self.writer is None:
            return
        self.totals.calls += 1
        self.totals.failed_calls += 1
        model = kwargs.get("model")
        self.writer.emit(
            "llm_call_failed",
            operation=self.operation,
            cognitive_mode=self.cognitive_mode,
            backend="llm",
            model=model,
            provider=self._provider_metadata(model),
            latency_ms=self._duration_ms(start_time, end_time),
            input={"messages": kwargs.get("messages")},
            output=dump_jsonable(response_obj),
            retry_count=None,
            error=redact_secrets(str(kwargs.get("exception") or kwargs.get("error") or "unknown")),
        )


def state_snapshot(state: Any) -> dict[str, Any]:
    session = state.session
    contexts: list[dict[str, Any]] = []
    for ctx in session.contexts:
        text = getattr(ctx, "text", None)
        doc = getattr(text, "doc", None)
        contexts.append(
            {
                "id": getattr(ctx, "id", None),
                "question": getattr(ctx, "question", None),
                "summary": getattr(ctx, "context", None),
                "relevance_score": getattr(ctx, "score", None),
                "chunk_name": getattr(text, "name", None),
                "evidence_text": getattr(text, "text", None),
                "docname": getattr(doc, "docname", None),
                "dockey": getattr(doc, "dockey", None),
                "citation": getattr(doc, "citation", None),
            }
        )
    return {
        "status_text": state.status,
        "paper_count": len(state.docs.docs),
        "contexts": contexts,
        "session": {
            "id": str(session.id),
            "question": session.question,
            "answer": session.answer,
            "raw_answer": session.raw_answer,
            "answer_reasoning": session.answer_reasoning,
            "has_successful_answer": session.has_successful_answer,
            "graded_answer": session.graded_answer,
            "references": session.references,
            "tool_history": session.tool_history,
            "token_counts": session.token_counts,
            "paperqa_reported_cost_usd": session.cost,
        },
    }


def attach_paperqa_callbacks(settings: Any, writer: TraceWriter, llm_logger: BaselineLLMLogger) -> None:
    agent_operation = (
        ("tool_argument_generation", "generation")
        if settings.agent.decision_backend == "jev"
        else ("tool_selection", "decision")
    )
    llm_logger.set_operation(*agent_operation)

    async def gather_init(state: Any) -> None:
        llm_logger.set_operation(
            "evidence_summary" if settings.answer.evidence_relevance_backend == "jev"
            else "evidence_processing",
            "generation" if settings.answer.evidence_relevance_backend == "jev" else "mixed",
        )
        writer.emit("gather_evidence_initialized", state=state_snapshot(state))

    async def gather_done(state: Any) -> None:
        writer.emit("gather_evidence_completed", state=state_snapshot(state))
        llm_logger.set_operation(*agent_operation)

    async def answer_init(state: Any) -> None:
        llm_logger.set_operation("final_answer", "generation")
        writer.emit("gen_answer_initialized", state=state_snapshot(state))

    async def answer_done(state: Any) -> None:
        writer.emit("gen_answer_completed", state=state_snapshot(state))
        llm_logger.set_operation(*agent_operation)

    settings.agent.callbacks = {
        "gather_evidence_initialized": [gather_init],
        "gather_evidence_completed": [gather_done],
        "gen_answer_initialized": [answer_init],
        "gen_answer_completed": [answer_done],
    }
