from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from aviary.core import ToolSelector
from paperqa.agents.main import run_agent
from paperqa.docs import Docs

from research.decision_model.tracing import JevTraceCollector

from .litqa2 import environment_class
from .paperqa_settings import ModelBundle
from .tracing import BaselineLLMLogger, TraceWriter, attach_paperqa_callbacks, state_snapshot
from .utils import (
    append_jsonl,
    dump_jsonable,
    git_value,
    redact_secrets,
    settings_fingerprint,
    utc_now,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    configuration: str
    mode: str
    benchmark: str | None
    split: str | None


async def run_one(
    *,
    query: Any,
    question_id: str,
    settings: Any,
    bundle: ModelBundle,
    identity: RunIdentity,
    llm_logger: BaselineLLMLogger,
    results_path: Path,
    trace_path: Path,
    sources: Any = None,
    pilot_gold_choice: str | None = None,
) -> dict[str, Any]:
    writer = TraceWriter(trace_path, identity.run_id, question_id)
    llm_logger.begin_question(writer, bundle)
    attach_paperqa_callbacks(settings, writer, llm_logger)
    jev_enabled = (
        settings.agent.decision_backend == "jev"
        or settings.answer.evidence_relevance_backend == "jev"
    )
    jev_trace = JevTraceCollector(writer, settings) if jev_enabled else None
    if jev_trace and settings.answer.evidence_relevance_backend == "jev":
        callbacks = dict(settings.agent.callbacks)
        callbacks["jev_relevance_result"] = [jev_trace.relevance_result]
        callbacks["jev_relevance_error"] = [jev_trace.relevance_error]
        settings.agent.callbacks = callbacks

    def cost_fields() -> dict[str, Any]:
        llm_observed_cost = round(llm_logger.totals.cost_usd, 12)
        llm_complete = llm_logger.totals.cost_observations == llm_logger.totals.calls
        llm_cost = llm_observed_cost if llm_complete else None
        jev_cost = round(jev_trace.totals.cost_usd, 12) if jev_trace else 0.0
        jev_complete = jev_trace.totals.cost_complete if jev_trace else True
        return {
            "llm_cost_usd": llm_cost,
            "llm_cost_observed_usd": llm_observed_cost,
            "llm_cost_complete": llm_complete,
            "jev_cost_usd": jev_cost,
            "jev_cost_complete": jev_complete,
            "total_cost_usd": (
                round(llm_cost + jev_cost, 12)
                if llm_cost is not None and jev_complete else None
            ),
            "jev_calls": jev_trace.totals.calls if jev_trace else 0,
            "failed_jev_calls": jev_trace.totals.failed_calls if jev_trace else 0,
            "jev_input_tokens": jev_trace.totals.input_tokens if jev_trace else 0,
            "jev_output_tokens": jev_trace.totals.output_tokens if jev_trace else 0,
            "jev_known_api_attempts": jev_trace.totals.known_api_attempts if jev_trace else 0,
            "jev_api_attempts_complete": jev_complete,
        }

    action_step = 0
    evaluation_result: dict[str, Any] = {}

    async def on_reset(state: Any) -> None:
        writer.emit("env_reset", state=state_snapshot(state))

    async def on_action(action: Any, _agent_state: Any) -> None:
        nonlocal action_step
        action_step += 1
        if jev_trace:
            jev_trace.step = action_step
        tool_calls = [
            {
                "selected_tool": call.function.name,
                "arguments": dump_jsonable(call.function.arguments),
            }
            for call in getattr(action, "tool_calls", ())
        ]
        writer.emit(
            "agent_action",
            step=action_step,
            cognitive_mode="decision",
            backend=settings.agent.decision_backend,
            tool_calls=tool_calls,
            action=dump_jsonable(action),
        )

    async def on_step(obs: list[Any], reward: float, done: bool, truncated: bool) -> None:
        writer.emit(
            "env_step",
            step=action_step,
            observations=dump_jsonable(obs),
            reward=reward,
            done=done,
            truncated=truncated,
        )

    async def evaluation_callback(evaluation: Any) -> None:
        value = getattr(evaluation, "value", str(evaluation))
        evaluation_result["evaluation"] = value
        writer.emit("evaluation", evaluation=value)

    writer.emit(
        "question_started",
        mode=identity.mode,
        benchmark=identity.benchmark,
        benchmark_split=identity.split,
        query=dump_jsonable(query),
        sources=dump_jsonable(sources),
        models=bundle.public_metadata(),
    )

    started = time.perf_counter()
    try:
        runner_kwargs: dict[str, Any] = {
            "on_env_reset_callback": on_reset,
            "on_agent_action_callback": on_action,
            "on_env_step_callback": on_step,
        }
        if jev_trace and settings.agent.decision_backend == "jev":
            runner_kwargs["on_jev_decision_callback"] = jev_trace.decision_result
            runner_kwargs["on_jev_decision_error_callback"] = jev_trace.decision_error
        if identity.benchmark == "LitQA2":
            runner_kwargs.update(
                env_class=environment_class(),
                sources=sources,
                session_id=UUID(question_id),
                evaluation_callback=evaluation_callback,
            )

        response = await run_agent(
            docs=Docs(),
            query=query,
            settings=settings,
            agent_type=ToolSelector,
            **runner_kwargs,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        session = response.session
        status = getattr(response.status, "value", str(response.status))
        evaluation = evaluation_result.get("evaluation")
        evaluation_text = str(evaluation).lower() if evaluation is not None else None
        correct = evaluation_text == "correct" if evaluation_text is not None else None
        pilot_prediction = None
        if pilot_gold_choice is not None:
            answer_match = re.search(
                r"(?:^|\n)\s*(?:Answer:\s*)?([A-Z])(?:[.):\s]|$)",
                session.answer or "",
                re.IGNORECASE,
            )
            pilot_prediction = answer_match.group(1).upper() if answer_match else None
            correct = (
                pilot_prediction == pilot_gold_choice
                if pilot_prediction is not None and status == "success"
                else None
            )

        record = {
            "run_id": identity.run_id,
            "question_id": question_id,
            "configuration": identity.configuration,
            "mode": identity.mode,
            "benchmark": identity.benchmark,
            "split": identity.split,
            "question": session.question,
            "final_answer": session.answer,
            "raw_answer": session.raw_answer,
            "answer_reasoning": session.answer_reasoning,
            "predicted_choice": (
                pilot_prediction if pilot_gold_choice is not None
                else session.graded_answer if identity.benchmark else None
            ),
            "gold_choice": (
                pilot_gold_choice if pilot_gold_choice is not None
                else getattr(query, "ideal_answer", None) if identity.benchmark else None
            ),
            "correct": correct,
            "letter_matches_gold": (
                pilot_prediction == pilot_gold_choice
                if pilot_gold_choice is not None and pilot_prediction is not None
                else None
            ),
            "evaluation": evaluation,
            "status": status,
            "has_successful_answer": session.has_successful_answer,
            "wall_clock_ms": elapsed_ms,
            **cost_fields(),
            "paperqa_reported_cost_usd": session.cost,
            "total_input_tokens": llm_logger.totals.input_tokens,
            "total_output_tokens": llm_logger.totals.output_tokens,
            "token_counts": session.token_counts,
            "llm_calls": llm_logger.totals.calls,
            "failed_llm_calls": llm_logger.totals.failed_calls,
            "tool_steps": action_step,
            "tool_history": session.tool_history,
            "config_md5": session.config_md5 or settings_fingerprint(settings),
            "commit_sha": git_value("rev-parse", "HEAD"),
            "models": bundle.public_metadata(),
            "trace_file": str(trace_path),
            "timestamp": utc_now(),
            "error": None,
        }
        append_jsonl(results_path, record)
        writer.emit("question_completed", result=record)
        return record
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        record = {
            "run_id": identity.run_id,
            "question_id": question_id,
            "configuration": identity.configuration,
            "mode": identity.mode,
            "benchmark": identity.benchmark,
            "split": identity.split,
            "question": getattr(query, "question", None) or str(query),
            "final_answer": None,
            "predicted_choice": None,
            "gold_choice": pilot_gold_choice or getattr(query, "ideal_answer", None),
            "correct": None,
            "status": "fail",
            "wall_clock_ms": elapsed_ms,
            **cost_fields(),
            "total_input_tokens": llm_logger.totals.input_tokens,
            "total_output_tokens": llm_logger.totals.output_tokens,
            "llm_calls": llm_logger.totals.calls,
            "failed_llm_calls": llm_logger.totals.failed_calls,
            "tool_steps": action_step,
            "config_md5": settings_fingerprint(settings),
            "commit_sha": git_value("rev-parse", "HEAD"),
            "models": bundle.public_metadata(),
            "trace_file": str(trace_path),
            "timestamp": utc_now(),
            "error": redact_secrets(f"{type(exc).__name__}: {exc}"),
        }
        append_jsonl(results_path, record)
        writer.emit("question_failed", result=record)
        logger.error("Question %s failed: %s", question_id, record["error"])
        return record


def failure_record(
    *,
    question_id: str,
    question: str,
    identity: RunIdentity,
    settings: Any,
    bundle: ModelBundle,
    results_path: Path,
    trace_path: Path,
    error: str,
) -> dict[str, Any]:
    writer = TraceWriter(trace_path, identity.run_id, question_id)
    record = {
        "run_id": identity.run_id,
        "question_id": question_id,
        "configuration": identity.configuration,
        "mode": identity.mode,
        "benchmark": identity.benchmark,
        "split": identity.split,
        "question": question,
        "status": "fail",
        "correct": None,
        "wall_clock_ms": 0.0,
        "llm_calls": 0,
        "config_md5": settings_fingerprint(settings),
        "commit_sha": git_value("rev-parse", "HEAD"),
        "models": bundle.public_metadata(),
        "trace_file": str(trace_path),
        "timestamp": utc_now(),
        "error": redact_secrets(error),
    }
    append_jsonl(results_path, record)
    writer.emit("preflight_failed", result=record)
    return record
