"""Offline checks for the opt-in Jev evidence path."""

import math
import unittest
from unittest.mock import patch

import httpx
from lmi import LLMResult

from paperqa.core import _map_fxn_summary, llm_parse_json
from paperqa.decision_models.jev import (
    JevRelevanceClient,
    JevRelevanceResult,
    build_relevance_request,
    parse_relevance_response,
)
from paperqa.docs import Docs
from paperqa.agents.tools import EnvironmentState, GatherEvidence
from paperqa.settings import AnswerSettings, Settings
from paperqa.types import Context, Doc, PQASession, Text


class FakeSummaryLLM:
    def __init__(self, response: str):
        self.response = response
        self.messages = None

    async def call_single(self, *, messages, **_kwargs):
        self.messages = messages
        return LLMResult(model="test-llm", date="", text=self.response, cost=0.02)


def sample_text() -> Text:
    return Text(
        name="paper-1 chunk-1",
        text="The trial reported a 20% reduction.",
        doc=Doc(docname="paper-1", dockey="paper-1", citation="Example, 2026"),
    )


def sample_jev_result() -> JevRelevanceResult:
    return JevRelevanceResult(
        score=8,
        raw_score=4.0,
        confidence=0.9,
        probabilities={str(i): float(i == 4) for i in range(6)},
        model="jev-1.13.0",
        latency_ms=20,
        input_tokens=100,
        output_tokens=10,
        cost_usd=0.001,
        retry_count=0,
    )


class JevEvidenceTest(unittest.IsolatedAsyncioTestCase):
    def test_jev_request_and_score_mapping(self):
        request = build_relevance_request("What happened?", "The trial...", "jev-latest")
        assert request["state"]["question"] == "What happened?"
        assert request["questions"]["evidence_relevance"]["type"] == "score"
        response = {
            "model": "jev-1.13.0",
            "answers": {
                "evidence_relevance": {
                    "type": "score",
                    "score": 3.75,
                    "confidence": 0.5,
                    "probabilities": {
                        "0": 0.0,
                        "1": 0.0,
                        "2": 0.0,
                        "3": 0.25,
                        "4": 0.75,
                        "5": 0.0,
                    },
                }
            },
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }
        result = parse_relevance_response(response, 20, 0, 10)
        assert result.score == 8
        assert result.cost_usd == 0.001

    async def test_client_sends_typed_request_and_records_retry(self):
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
                        "evidence_relevance": {
                            "type": "score",
                            "score": 4.0,
                            "confidence": 1.0,
                            "probabilities": {str(i): float(i == 4) for i in range(6)},
                        }
                    },
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                },
            )

        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(respond)

        async def no_sleep(_seconds):  # noqa: RUF029
            return None

        with (
            patch(
                "paperqa.decision_models.jev.httpx.AsyncClient",
                side_effect=lambda **kwargs: real_client(transport=transport, **kwargs),
            ),
            patch("paperqa.decision_models.jev.asyncio.sleep", no_sleep),
        ):
            client = JevRelevanceClient(
                model="jev-1.13.0",
                input_price_usd_per_million=10,
                api_key="test-key",
            )
            result = await client.score("Question", "Evidence")
            await client.aclose()
        assert result.retry_count == 1
        assert result.score == 8
        assert len(calls) == 2
        assert calls[0].url.path == "/v1/systemone"
        assert calls[0].headers["Authorization"] == "Bearer test-key"

    async def test_default_core_path_keeps_llm_relevance_score(self):
        llm = FakeSummaryLLM('{"summary":"Twenty percent reduction.","relevance_score":6}')
        context, _ = await _map_fxn_summary(
            text=sample_text(),
            question="What did the trial report?",
            summary_llm_model=llm,
            prompt_templates=("{text}", "Return a summary and score."),
            parser=llm_parse_json,
        )
        assert context.score == 6
        assert context.context == "Twenty percent reduction."

    async def test_core_uses_jev_score_and_keeps_llm_summary(self):
        llm = FakeSummaryLLM('{"summary":"Twenty percent reduction."}')

        async def score(_question, _evidence):  # noqa: RUF029
            return sample_jev_result()

        context, llm_results = await _map_fxn_summary(
            text=sample_text(),
            question="What did the trial report?",
            summary_llm_model=llm,
            prompt_templates=("{text}", "Return a summary."),
            parser=llm_parse_json,
            relevance_scorer=score,
        )
        assert context.context == "Twenty percent reduction."
        assert context.score == 8
        assert len(llm_results) == 1

    async def test_docs_opt_in_adds_jev_cost_without_changing_baseline_default(self):
        assert Settings().answer.evidence_relevance_backend == "llm"
        text = sample_text()
        docs = Docs(docs={text.doc.dockey: text.doc}, texts=[text])
        llm = FakeSummaryLLM('{"summary":"Twenty percent reduction."}')
        settings = Settings(
            answer=AnswerSettings(
                evidence_relevance_backend="jev",
                evidence_retrieval=False,
                jev_input_price_usd_per_million=10,
            )
        )
        traced = []
        with (
            patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-key"}),
            patch.object(JevRelevanceClient, "score", return_value=sample_jev_result()),
        ):
            session = await docs.aget_evidence(
                "What did the trial report?",
                settings=settings,
                embedding_model=object(),
                summary_llm_model=llm,
                relevance_trace_callback=lambda *_args: traced.append(_args),
            )
        assert len(session.contexts) == 1
        assert session.contexts[0].score == 8
        assert math.isclose(session.cost, 0.021)
        assert len(traced) == 1
        assert traced[0][0].text.text == text.text
        assert traced[0][1].input_tokens == 100
        assert session.token_counts["jev-1.13.0"] == [100, 10]
        assert "Do not provide a relevance score" in llm.messages[0].content

    async def test_gather_tool_forwards_jev_result_and_error_callbacks(self):
        text = sample_text()
        settings = Settings()
        received = []
        settings.agent.callbacks = {
            "jev_relevance_result": [lambda context, result: received.append(("ok", context, result))],
            "jev_relevance_error": [lambda *args: received.append(("error", *args))],
        }
        state = EnvironmentState(
            docs=Docs(docs={text.doc.dockey: text.doc}),
            session=PQASession(question="What happened?"),
        )
        tool = GatherEvidence.model_construct(
            settings=settings, summary_llm_model=None, embedding_model=None,
            partitioning_fn=None,
        )

        async def fake_evidence(_docs, **kwargs):
            context = Context(context="A summary.", question="What happened?", text=text, score=8)
            kwargs["relevance_trace_callback"](context, sample_jev_result())
            kwargs["relevance_error_callback"](
                text, "What happened?", text.text, RuntimeError("failed"), 3.0
            )
            return kwargs["query"]

        with patch.object(Docs, "aget_evidence", fake_evidence):
            await tool.gather_evidence("What happened?", state)
        assert [item[0] for item in received] == ["ok", "error"]
        assert received[0][1].text.name == text.name
        assert received[1][2] == "What happened?"


if __name__ == "__main__":
    unittest.main()
