"""Offline checks for Jev records in the shared baseline logger."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from paperqa.agents.models import AgentStatus
from paperqa.decision_models.jev import JevRelevanceResult
from paperqa.decision_models.jev_decisions import JevDecisionResult
from paperqa.settings import AgentSettings, AnswerSettings, Settings
from paperqa.types import PQASession
from research.baseline.executor import RunIdentity, run_one
from research.baseline.paperqa_settings import ModelBundle
from research.baseline.providers.base import ResolvedModelTarget
from research.baseline.tracing import BaselineLLMLogger
from research.baseline.tracing import TraceWriter
from research.decision_model.tracing import JevTraceCollector


def bundle() -> ModelBundle:
    target = ResolvedModelTarget(
        target_name="test", provider_name="groq", adapter_name="litellm",
        requested_model="openai/gpt-oss-120b",
        litellm_model="groq/openai/gpt-oss-120b", temperature=0.0,
        litellm_params={"model": "groq/openai/gpt-oss-120b"}, router_kwargs={},
    )
    return ModelBundle.homogeneous(target)


class JevLoggingIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_termination_and_relevance_timeout_have_separate_records(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            settings = Settings(
                agent=AgentSettings(decision_backend="jev"),
                answer=AnswerSettings(evidence_relevance_backend="jev"),
            )
            collector = JevTraceCollector(TraceWriter(trace, "run", "question"), settings)
            await collector.decision_result(
                JevDecisionResult(
                    choice="complete_success", confidence=0.95,
                    probabilities={"complete_success": 1.0}, model="jev-1.13.0",
                    latency_ms=4.0, input_tokens=20, output_tokens=1,
                    cost_usd=0.0001, retry_count=0,
                    request={"state": {"messages": ["answer ready"]}},
                ),
                None,
            )
            text = SimpleNamespace(
                name="chunk", doc=SimpleNamespace(docname="paper", dockey="key")
            )
            collector.relevance_error(
                text, "Question", "Evidence", TimeoutError("timed out"), 1000.0
            )
            events = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual(events[0]["operation"], "termination")
            self.assertEqual(events[0]["output"]["choice"], "complete_success")
            self.assertEqual(events[1]["operation"], "evidence_relevance")
            self.assertEqual(events[1]["input"]["state"]["evidence_text"], "Evidence")
            self.assertTrue(events[1]["timeout"])
            self.assertEqual(collector.totals.calls, 2)
            self.assertFalse(collector.totals.cost_complete)

    async def test_runner_records_decision_relevance_and_question_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.jsonl"
            trace = Path(directory) / "trace.jsonl"
            settings = Settings(
                agent=AgentSettings(decision_backend="jev"),
                answer=AnswerSettings(evidence_relevance_backend="jev"),
            )
            logger = BaselineLLMLogger()

            async def fake_run_agent(*, docs, query, settings: Settings, **kwargs):
                run_settings = settings
                assert "jev_relevance_result" in run_settings.agent.callbacks
                decision = JevDecisionResult(
                    choice="paper_search", confidence=0.8,
                    probabilities={"paper_search": 1.0}, model="jev-1.13.0",
                    latency_ms=12.0, input_tokens=100, output_tokens=2,
                    cost_usd=0.0002, retry_count=1,
                    request={"state": {"messages": ["question"]}},
                )
                await kwargs["on_jev_decision_callback"](decision, None)
                await kwargs["on_agent_action_callback"]({"tool": "paper_search"}, None)
                context = SimpleNamespace(
                    id="chunk-1", question="What happened?", context="A summary.", score=8,
                    text=SimpleNamespace(
                        name="paper chunk", text="Evidence text.",
                        doc=SimpleNamespace(docname="paper", dockey="paper-key"),
                    ),
                )
                relevance = JevRelevanceResult(
                    score=8, raw_score=4.0, confidence=0.9,
                    probabilities={str(i): float(i == 4) for i in range(6)},
                    model="jev-1.13.0", latency_ms=20.0,
                    input_tokens=200, output_tokens=3,
                    cost_usd=0.0005, retry_count=0,
                )
                run_settings.agent.callbacks["jev_relevance_result"][0](context, relevance)
                logger.totals.calls = 2
                logger.totals.cost_observations = 2
                logger.totals.cost_usd = 0.01
                return SimpleNamespace(
                    session=PQASession(question="What happened?", answer="B",
                                       has_successful_answer=True),
                    status=AgentStatus.SUCCESS,
                )

            with patch("research.baseline.executor.run_agent", side_effect=fake_run_agent):
                record = await run_one(
                    query="What happened?", question_id="user-test", settings=settings,
                    bundle=bundle(), identity=RunIdentity("run", "heterogeneous_static", "user", None, None),
                    llm_logger=logger, results_path=results, trace_path=trace,
                    pilot_gold_choice="B",
                )

            events = [json.loads(line) for line in trace.read_text().splitlines()]
            jev_calls = [event for event in events if event["event"] == "jev_call"]
            self.assertEqual([event["operation"] for event in jev_calls],
                             ["tool_selection", "evidence_relevance"])
            self.assertEqual(jev_calls[0]["input"]["state"]["messages"], ["question"])
            self.assertEqual(jev_calls[1]["input"]["state"]["evidence_text"], "Evidence text.")
            self.assertEqual(jev_calls[1]["output"]["summary"], "A summary.")
            self.assertEqual(record["jev_calls"], 2)
            self.assertEqual(record["predicted_choice"], "B")
            self.assertTrue(record["correct"])
            self.assertEqual(record["jev_input_tokens"], 300)
            self.assertAlmostEqual(record["jev_cost_usd"], 0.0007)
            self.assertAlmostEqual(record["total_cost_usd"], 0.0107)
            self.assertEqual(json.loads(results.read_text())["jev_calls"], 2)

    async def test_runner_keeps_failed_jev_call_and_unknown_total_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.jsonl"
            trace = Path(directory) / "trace.jsonl"
            settings = Settings(agent=AgentSettings(decision_backend="jev"))

            async def fake_run_agent(*, docs, query, settings, **kwargs):
                await kwargs["on_jev_decision_error_callback"](
                    {"state": {"messages": []}}, RuntimeError("API failed"), 5.0, None
                )
                raise RuntimeError("trajectory failed")

            with patch("research.baseline.executor.run_agent", side_effect=fake_run_agent):
                record = await run_one(
                    query="Question", question_id="user-test", settings=settings,
                    bundle=bundle(), identity=RunIdentity("run", "heterogeneous_static", "user", None, None),
                    llm_logger=BaselineLLMLogger(), results_path=results, trace_path=trace,
                )
            events = [json.loads(line) for line in trace.read_text().splitlines()]
            failures = [event for event in events if event["event"] == "jev_call_failed"]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["error"], "RuntimeError: API failed")
            self.assertEqual(record["jev_calls"], 1)
            self.assertEqual(record["failed_jev_calls"], 1)
            self.assertFalse(record["jev_cost_complete"])
            self.assertIsNone(record["total_cost_usd"])
