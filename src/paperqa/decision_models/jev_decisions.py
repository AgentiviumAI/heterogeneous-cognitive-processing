"""Opt-in Jev tool choice with LLM-generated tool arguments."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any

import httpx
import litellm
from aviary.core import MalformedMessageError, Message, Tool, ToolCall, ToolRequestMessage
from lmi import LiteLLMModel

from .jev import JEV_ENDPOINT

QUESTION_ID = "next_action"
COMPLETE_SUCCESS = "complete_success"
COMPLETE_UNSURE = "complete_unsure"


@dataclass(frozen=True)
class JevDecisionResult:
    choice: str
    confidence: float
    probabilities: dict[str, float]
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    retry_count: int
    request: dict[str, Any] | None = None


def tool_criteria(tools: list[Tool]) -> dict[str, str]:
    """Give Jev every available action, with completion certainty made explicit."""
    criteria = {
        tool.info.name: tool.info.description for tool in tools if tool.info.name != "complete"
    }
    if any(tool.info.name == "complete" for tool in tools):
        criteria[COMPLETE_SUCCESS] = (
            "Stop now because the generated answer addresses the question."
        )
        criteria[COMPLETE_UNSURE] = "Stop now because no reliable answer can be given."
    return criteria


def build_decision_request(
    messages: list[Message], tools: list[Tool], model: str
) -> dict[str, Any]:
    criteria = tool_criteria(tools)
    if not criteria:
        raise ValueError("No tools are available for Jev selection")
    return {
        "model": model,
        "state": {
            "messages": [
                message.model_dump(mode="json", exclude_none=True, exclude={"info"})
                for message in messages
            ]
        },
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": (
                    "Which single action should PaperQA take next to answer the user's "
                    "question? Use the message history and tool results. Search when "
                    "more papers are needed; gather evidence after papers are found; "
                    "generate an answer when evidence is sufficient. Choose a completion "
                    "option only when the workflow should stop."
                ),
                "criteria": criteria,
            }
        },
    }


def parse_decision_response(
    body: dict[str, Any], criteria: dict[str, str], latency_ms: float, retries: int, price: float
) -> JevDecisionResult:
    try:
        answer = body["answers"][QUESTION_ID]
        usage = body["usage"]
        choice = answer["choice"]
        confidence = float(answer["confidence"])
        probabilities = {key: float(value) for key, value in answer["probabilities"].items()}
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        model = body["model"]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid Jev decision response") from exc
    invalid_choice = (
        answer.get("type") != "choice"
        or not isinstance(choice, str)
        or choice not in criteria
        or set(probabilities) != set(criteria)
        or not isinstance(model, str)
    )
    invalid_confidence = not math.isfinite(confidence) or not 0 <= confidence <= 1
    invalid_probabilities = not all(
        math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()
    ) or not math.isclose(sum(probabilities.values()), 1, abs_tol=0.02)
    invalid_usage = (
        not isinstance(input_tokens, int)
        or not isinstance(output_tokens, int)
        or input_tokens < 0
        or output_tokens < 0
    )
    if invalid_choice or invalid_confidence or invalid_probabilities or invalid_usage:
        raise ValueError("Invalid Jev decision response")
    return JevDecisionResult(
        choice=choice,
        confidence=confidence,
        probabilities=probabilities,
        model=model,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=input_tokens * price / 1_000_000,
        retry_count=retries,
    )


class JevDecisionClient:
    def __init__(
        self,
        *,
        model: str,
        input_price_usd_per_million: float,
        api_key: str | None = None,
        timeout_s: float = 30,
        retries: int = 2,
    ) -> None:
        self.model = model
        self.price = input_price_usd_per_million
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        self.retries = retries
        if not self.api_key:
            raise ValueError("TYPESAFE_API_KEY is required for Jev tool selection")
        if self.price < 0:
            raise ValueError("Jev input price must be non-negative")
        self._http = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def choose(self, messages: list[Message], tools: list[Tool]) -> JevDecisionResult:
        payload = build_decision_request(messages, tools, self.model)
        criteria = payload["questions"][QUESTION_ID]["criteria"]
        start = perf_counter()
        for attempt in range(self.retries + 1):
            try:
                response = await self._http.post(
                    JEV_ENDPOINT,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise TypeError("Invalid Jev JSON response")
                return replace(
                    parse_decision_response(
                        body, criteria, (perf_counter() - start) * 1000, attempt, self.price
                    ),
                    request=payload,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {429, 529} or attempt == self.retries:
                    raise
            except httpx.RequestError:
                if attempt == self.retries:
                    raise
            await asyncio.sleep(0.5 * (2**attempt))
        raise AssertionError("unreachable")


class JevToolSelector:
    """Return the same ToolRequestMessage shape as PaperQA's LLM selector."""

    def __init__(
        self,
        llm: LiteLLMModel,
        client: JevDecisionClient,
        on_decision: Callable[[JevDecisionResult], Awaitable[None]] | None = None,
        on_error: Callable[[dict[str, Any], Exception, float], Awaitable[None]] | None = None,
    ):
        self.llm = llm
        self.client = client
        self.on_decision = on_decision
        self.on_error = on_error

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __call__(self, messages: list[Message], tools: list[Tool]) -> ToolRequestMessage:
        started = perf_counter()
        try:
            decision = await self.client.choose(messages, tools)
        except Exception as exc:
            if self.on_error is not None:
                await self.on_error(
                    build_decision_request(messages, tools, self.client.model),
                    exc,
                    (perf_counter() - started) * 1000,
                )
            raise
        if self.on_decision is not None:
            await self.on_decision(decision)
        if decision.choice in {COMPLETE_SUCCESS, COMPLETE_UNSURE}:
            complete = next(tool for tool in tools if tool.info.name == "complete")
            action = ToolRequestMessage(
                tool_calls=[
                    ToolCall.from_tool(
                        complete,
                        has_successful_answer=decision.choice == COMPLETE_SUCCESS,
                    )
                ]
            )
        else:
            selected = next(tool for tool in tools if tool.info.name == decision.choice)
            if decision.choice in {"gen_answer", "reset"}:
                action = ToolRequestMessage(tool_calls=[ToolCall.from_tool(selected)])
            elif getattr(self.llm, "name", "").startswith("gemini/gemma-"):
                action = await self._generate_arguments_as_json(messages, selected)
            else:
                try:
                    action = await self.llm.select_tool(
                        messages=messages, tools=[selected], tool_choice=selected
                    )
                except litellm.BadRequestError as exc:
                    if "tool_use_failed" not in str(exc):
                        raise
                    action = await self._generate_arguments_as_json(messages, selected)
            if (
                len(action.tool_calls) != 1
                or action.tool_calls[0].function.name != decision.choice
            ):
                raise MalformedMessageError(
                    "LLM did not generate arguments for Jev's selected tool"
                )
        decision_info = asdict(decision)
        decision_info.pop("request", None)
        action.info = {**(action.info or {}), "jev_decision": decision_info}
        return action

    async def _generate_arguments_as_json(
        self, messages: list[Message], selected: Tool
    ) -> ToolRequestMessage:
        """Recover when a provider rejects a forced tool call."""
        prompt = (
            "Generate only the arguments for the selected PaperQA tool. "
            "Return one JSON object with exactly the required argument names. "
            "Do not choose or name a different tool.\n"
            f"Selected tool: {selected.info.name}\n"
            f"Schema: {json.dumps(selected.info.model_dump(mode='json')['parameters'])}\n"
            "Recent conversation:\n"
            + "\n".join(str(message.content) for message in messages[-8:])
        )
        response = await self.llm.call_single(messages=[Message(role="user", content=prompt)])
        match = re.search(r"\{.*\}", str(response.text), re.DOTALL)
        if match is None:
            raise MalformedMessageError("LLM did not return JSON tool arguments")
        try:
            arguments = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise MalformedMessageError("LLM returned invalid JSON tool arguments") from exc
        if not isinstance(arguments, dict):
            raise MalformedMessageError("LLM tool arguments must be a JSON object")
        return ToolRequestMessage(tool_calls=[ToolCall.from_tool(selected, **arguments)])
