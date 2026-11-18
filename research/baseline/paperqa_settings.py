from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from paperqa.settings import (
    AgentSettings,
    IndexSettings,
    MultimodalOptions,
    Settings,
)

from .providers import ResolvedModelTarget


@dataclass(frozen=True)
class ModelBundle:
    """Resolved models for the three PaperQA LLM roles.

    Homogeneous baseline uses the same target for all roles. Keeping the roles
    explicit lets later experiments replace one role without changing runner code.
    """

    answer: ResolvedModelTarget
    summary: ResolvedModelTarget
    agent: ResolvedModelTarget

    @classmethod
    def homogeneous(cls, target: ResolvedModelTarget) -> "ModelBundle":
        return cls(answer=target, summary=target, agent=target)

    def public_metadata(self) -> dict[str, Any]:
        return {
            "answer": self.answer.public_metadata(),
            "summary": self.summary.public_metadata(),
            "agent": self.agent.public_metadata(),
            "homogeneous": (
                self.answer.litellm_model
                == self.summary.litellm_model
                == self.agent.litellm_model
            ),
        }


@dataclass(frozen=True)
class PaperQARunOptions:
    paper_dir: str
    index_name: str
    manifest: str | None = None
    embedding: str = "st-multi-qa-MiniLM-L6-cos-v1"
    search_count: int = 8
    agent_evidence_n: int = 1
    max_timesteps: int | None = None
    timeout: float = 500.0
    retries: int = 3
    index_concurrency: int = 1
    index_batch_size: int = 1
    rebuild_index: bool = False
    verbosity: int = 1

    metadata_mode: str = "upstream"
    multimodal_mode: str = "enrich"
    max_concurrent_requests: int = 1

def build_settings(bundle: ModelBundle, options: PaperQARunOptions) -> Settings:
    settings = Settings(
        llm=bundle.answer.litellm_model,
        llm_config=bundle.answer.router_config(retries=options.retries),
        summary_llm=bundle.summary.litellm_model,
        summary_llm_config=bundle.summary.router_config(retries=options.retries),
        embedding=options.embedding,
        temperature=bundle.answer.temperature,
        verbosity=options.verbosity,
        agent=AgentSettings(
            agent_llm=bundle.agent.litellm_model,
            agent_llm_config=bundle.agent.router_config(
                retries=options.retries
            ),
            agent_type="ToolSelector",
            search_count=options.search_count,
            agent_evidence_n=options.agent_evidence_n,
            max_timesteps=options.max_timesteps,
            timeout=options.timeout,
            index=IndexSettings(
                name=options.index_name,
                paper_directory=options.paper_dir,
                manifest_file=options.manifest,
                concurrency=options.index_concurrency,
                batch_size=options.index_batch_size,
            ),
            rebuild_index=options.rebuild_index,
        ),
    )

    settings.parsing.use_doc_details = options.metadata_mode == "upstream"
    settings.answer.max_concurrent_requests = options.max_concurrent_requests

    settings.parsing.multimodal = {
        "enrich": MultimodalOptions.ON_WITH_ENRICHMENT,
        "parse-only": MultimodalOptions.ON_WITHOUT_ENRICHMENT,
        "off": MultimodalOptions.OFF,
    }[options.multimodal_mode]

    return settings
