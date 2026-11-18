"""Check that the baseline runner can opt into Jev after branch integration."""

import os
import unittest
from unittest.mock import patch

from aviary.core import ToolSelector

from paperqa.decision_models.jev_decisions import JevToolSelector
from research.baseline.paperqa_settings import ModelBundle, PaperQARunOptions, build_settings
from research.baseline.cli import build_parser
from research.baseline.providers import ModelTargetSpec, ProviderRegistry, ProviderSpec


class BaselineJevIntegrationTest(unittest.IsolatedAsyncioTestCase):
    def test_cli_exposes_opt_in_jev_backends(self):
        args = build_parser().parse_args([
            "user", "--provider", "groq", "--model", "openai/gpt-oss-120b",
            "--paper-dir", "/tmp/papers", "--index-name", "pilot",
            "--question", "What happened?", "--decision-backend", "jev",
            "--evidence-relevance-backend", "jev",
        ])
        self.assertEqual(args.decision_backend, "jev")
        self.assertEqual(args.evidence_relevance_backend, "jev")

    async def test_baseline_settings_can_select_jev_without_changing_model_roles(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-only", "TYPESAFE_API_KEY": "test-only"}):
            provider = ProviderSpec(
                name="groq", adapter="litellm", litellm_prefix="groq", api_key_env="GROQ_API_KEY"
            )
            target = ModelTargetSpec(
                name="pilot", provider="groq", model="openai/gpt-oss-120b"
            )
            resolved = ProviderRegistry().resolve(provider, target)
            settings = build_settings(
                ModelBundle.homogeneous(resolved),
                PaperQARunOptions(
                    paper_dir="/tmp/litqa2-pilot", index_name="litqa2-pilot",
                    metadata_mode="off", multimodal_mode="off",
                ),
            )
            self.assertIs(type(settings.make_aviary_tool_selector(ToolSelector)), ToolSelector)
            self.assertEqual(settings.answer.evidence_relevance_backend, "llm")

            settings.agent.decision_backend = "jev"
            settings.answer.evidence_relevance_backend = "jev"
            selector = settings.make_aviary_tool_selector(ToolSelector)
            try:
                self.assertIsInstance(selector, JevToolSelector)
                self.assertEqual(settings.llm, settings.summary_llm)
                self.assertEqual(settings.llm, settings.agent.agent_llm)
            finally:
                await selector.aclose()
