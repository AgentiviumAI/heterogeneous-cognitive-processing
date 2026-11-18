"""Offline contract checks for Jev tool and stop decisions."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from aviary.core import Message, Tool, ToolCall, ToolRequestMessage

from paperqa.agents.main import run_aviary_agent
from paperqa.agents.tools import EnvironmentState
from paperqa.decision_models.jev_decisions import (
    JevDecisionClient,
    JevDecisionResult,
    JevToolSelector,
    build_decision_request,
    parse_decision_response,
)
from paperqa.docs import Docs
from paperqa.settings import Settings
from paperqa.types import PQASession


def paper_search(query: str) -> str:
    """Search papers for a topic.

    Args:
        query: Search phrase.
    """
    return query


def complete(has_successful_answer: bool) -> str:
    """Stop the workflow.

    Args:
        has_successful_answer: Whether the answer is reliable.
    """
    return str(has_successful_answer)


TOOLS = [Tool.from_function(paper_search), Tool.from_function(complete)]
MESSAGES = [Message(role="user", content="Find the paper")]


def result(choice: str) -> JevDecisionResult:
    return JevDecisionResult(
        choice=choice,
        confidence=1,
        probabilities={"paper_search": 0, "complete_success": 0, "complete_unsure": 0},
        model="jev-1.13.0",
        latency_ms=20,
        input_tokens=100,
        output_tokens=8,
        cost_usd=0.0000042,
        retry_count=0,
    )


class FakeClient:
    def __init__(self, choice: str):
        self.choice = choice
        self.closed = False

    async def choose(self, _messages, _tools):
        return result(self.choice)

    async def aclose(self):
        self.closed = True


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def select_tool(self, *, messages, tools, tool_choice):
        self.calls += 1
        assert messages == MESSAGES
        assert tools == [TOOLS[0]]
        return ToolRequestMessage(
            tool_calls=[ToolCall.from_tool(tool_choice, query="bacterial resistance")],
            info={"model": "test-llm", "usage": (12, 4)},
        )


class FakeEnvironment:
    def __init__(self, query, _settings, docs):
        self.state = EnvironmentState(docs=docs, session=PQASession(question=query))
        self.state.session.tool_history.append(["gen_answer"])

    async def reset(self):
        return MESSAGES, TOOLS

    async def step(self, action):
        self.state.record_action(action)
        self.state.session.has_successful_answer = True
        return [], 0, True, False


class JevDecisionsTest(unittest.IsolatedAsyncioTestCase):
    async def test_json_argument_fallback(self):
        class TextLLM:
            async def call_single(self, *, messages):
                assert "paper_search" in messages[0].content
                return SimpleNamespace(text='{"query": "bacterial resistance"}')

        selector = JevToolSelector(TextLLM(), FakeClient("paper_search"))
        action = await selector._generate_arguments_as_json(MESSAGES, TOOLS[0])
        assert action.tool_calls[0].function.name == "paper_search"

    async def test_client_sends_choice_request_and_retries_rate_limit(self):
        calls = []

        def respond(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(429, json={"error": "rate limit"})
            return httpx.Response(
                200,
                json={
                    "model": "jev-1.13.0",
                    "answers": {
                        "next_action": {
                            "type": "choice",
                            "choice": "paper_search",
                            "confidence": 1,
                            "probabilities": {
                                "paper_search": 1,
                                "complete_success": 0,
                                "complete_unsure": 0,
                            },
                        }
                    },
                    "usage": {"input_tokens": 100, "output_tokens": 8},
                },
            )

        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(respond)

        async def no_sleep(_seconds):  # noqa: RUF029
            return None

        with (
            patch(
                "paperqa.decision_models.jev_decisions.httpx.AsyncClient",
                side_effect=lambda **kwargs: real_client(transport=transport, **kwargs),
            ),
            patch("paperqa.decision_models.jev_decisions.asyncio.sleep", no_sleep),
        ):
            client = JevDecisionClient(
                model="jev-1.13.0", input_price_usd_per_million=0.042, api_key="test-key"
            )
            decision = await client.choose(MESSAGES, TOOLS)
            await client.aclose()
        assert decision.retry_count == 1
        assert decision.choice == "paper_search"
        assert decision.request["state"]["messages"]
        assert calls[0].url.path == "/v1/systemone"
        assert calls[0].headers["Authorization"] == "Bearer test-key"

    def test_request_and_response_contract(self):
        payload = build_decision_request(MESSAGES, TOOLS, "jev-1.13.0")
        criteria = payload["questions"]["next_action"]["criteria"]
        assert set(criteria) == {"paper_search", "complete_success", "complete_unsure"}
        response = {
            "model": "jev-1.13.0",
            "answers": {
                "next_action": {
                    "type": "choice",
                    "choice": "paper_search",
                    "confidence": 0.8,
                    "probabilities": {
                        "paper_search": 0.9,
                        "complete_success": 0.05,
                        "complete_unsure": 0.05,
                    },
                }
            },
            "usage": {"input_tokens": 100, "output_tokens": 8},
        }
        parsed = parse_decision_response(response, criteria, 20, 0, 0.042)
        assert parsed.choice == "paper_search"
        assert math.isclose(parsed.cost_usd, 0.0000042)
        response["answers"]["next_action"]["choice"] = "unknown"
        with self.assertRaises(ValueError):  # noqa: PT027
            parse_decision_response(response, criteria, 20, 0, 0.042)

    async def test_jev_selects_tool_llm_only_generates_arguments(self):
        llm = FakeLLM()
        client = FakeClient("paper_search")
        selector = JevToolSelector(llm, client)
        action = await selector(MESSAGES, TOOLS)
        assert action.tool_calls[0].function.name == "paper_search"
        assert action.tool_calls[0].function.arguments == {"query": "bacterial resistance"}
        assert action.info["jev_decision"]["choice"] == "paper_search"
        assert action.info["model"] == "test-llm"
        assert llm.calls == 1
        await selector.aclose()
        assert client.closed

    async def test_jev_request_error_reaches_trace_callback(self):
        class FailingClient:
            model = "jev-1.13.0"

            async def choose(self, _messages, _tools):
                raise RuntimeError("Jev unavailable")

            async def aclose(self):
                pass

        errors = []

        async def on_error(request, error, latency_ms):
            errors.append((request, error, latency_ms))

        selector = JevToolSelector(FakeLLM(), FailingClient(), on_error=on_error)
        with self.assertRaisesRegex(RuntimeError, "Jev unavailable"):
            await selector(MESSAGES, TOOLS)
        assert len(errors) == 1
        assert errors[0][0]["questions"]["next_action"]["type"] == "choice"
        assert errors[0][2] >= 0

    async def test_jev_complete_maps_to_existing_tool_without_llm(self):
        for choice, expected in (("complete_success", True), ("complete_unsure", False)):
            llm = FakeLLM()
            selector = JevToolSelector(llm, FakeClient(choice))
            action = await selector(MESSAGES, TOOLS)
            assert action.tool_calls[0].function.name == "complete"
            assert action.tool_calls[0].function.arguments == {"has_successful_answer": expected}
            assert llm.calls == 0

    async def test_agent_counts_jev_usage_and_exposes_decision_trace(self):
        selector = JevToolSelector(FakeLLM(), FakeClient("complete_success"))
        traced = []
        decisions = []

        async def on_action(action, _state):  # noqa: RUF029
            traced.append(action.info["jev_decision"])

        async def on_decision(decision, _state):  # noqa: RUF029
            decisions.append(decision)

        with patch.object(Settings, "adjust_tools_for_agent_llm", return_value=None):
            session, _status = await run_aviary_agent(
                "Find the paper",
                Settings(),
                Docs(),
                selector,
                env_class=FakeEnvironment,
                on_agent_action_callback=on_action,
                on_jev_decision_callback=on_decision,
            )
        assert math.isclose(session.cost, 0.0000042)
        assert session.token_counts["jev-1.13.0"] == [100, 8]
        assert traced[0]["choice"] == "complete_success"
        assert decisions[0].choice == "complete_success"
        assert selector.client.closed


if __name__ == "__main__":
    unittest.main()
