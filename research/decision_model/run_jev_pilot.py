"""Run the five prepared LitQA2 integration questions through Jev only."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import string
import sys
from datetime import datetime
from pathlib import Path

import litellm
from dotenv import load_dotenv

from paperqa.agents.search import get_directory_index
from paperqa.settings import AgentSettings, AnswerSettings, IndexSettings, ParsingSettings, Settings
from research.baseline.executor import RunIdentity, run_one
from research.baseline.paperqa_settings import ModelBundle
from research.baseline.providers.base import ResolvedModelTarget
from research.baseline.tracing import BaselineLLMLogger
from research.baseline.utils import RedactedStream, utc_now

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = Path("/tmp/litqa2-pilot-5-fulltext/questions.jsonl")
DEFAULT_ANSWERS = Path("/tmp/litqa2-pilot-5-fulltext/answer_key.jsonl")
DEFAULT_CORPUS = Path("/tmp/hcp-litqa2-fulltext-corpus")


def prompt_for(question: dict) -> str:
    choices = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(string.ascii_uppercase, question["choices"])
    )
    return (
        f"{question['question']}\n\n{choices}\n\n"
        "Choose the best answer. Begin your final answer with its letter, "
        "followed by a short explanation and citation."
    )


def settings_for(corpus: Path, model: str, max_timesteps: int, timeout: float) -> Settings:
    return Settings(
        llm=model,
        summary_llm=model,
        embedding="sparse",
        parsing=ParsingSettings(use_doc_details=False, multimodal=False),
        answer=AnswerSettings(
            evidence_relevance_backend="jev", evidence_k=3, max_concurrent_requests=1
        ),
        agent=AgentSettings(
            agent_llm=model,
            decision_backend="jev",
            index=IndexSettings(
                name="litqa2-five-fulltext",
                paper_directory=corpus,
                manifest_file="manifest.csv",
                index_directory=corpus.parent / "hcp-litqa2-fulltext-index",
            ),
            max_timesteps=max_timesteps,
            timeout=timeout,
            search_count=2,
        ),
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--answer-key", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", default="groq/openai/gpt-oss-120b")
    parser.add_argument("--extra-env-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/jev-pilot"))
    parser.add_argument("--run-id", default=datetime.now().strftime("jev_pilot_%Y%m%d_%H%M%S"))
    parser.add_argument("--max-timesteps", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--sleep-between-questions", type=float, default=0.0)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if args.extra_env_file:
        load_dotenv(args.extra_env_file)
    sys.stderr = RedactedStream(sys.stderr)
    if args.model.startswith("groq/"):
        provider, key_name = "groq", "GROQ_API_KEY"
    elif args.model.startswith("gemini/"):
        provider, key_name = "gemini", "GEMINI_API_KEY"
    else:
        raise ValueError("--model must start with groq/ or gemini/")
    for name in (key_name, "TYPESAFE_API_KEY"):
        if not os.getenv(name):
            raise SystemExit(f"{name} is required")
    questions = [json.loads(line) for line in args.questions.read_text().splitlines()]
    answers = {
        row["question_id"]: row["gold_choice"]
        for line in args.answer_key.read_text().splitlines()
        if (row := json.loads(line))
    }
    if len(questions) != 5 or set(answers) != {q["question_id"] for q in questions}:
        raise ValueError("Expected the same five question IDs in questions and answer key")
    if not 1 <= args.limit <= 5:
        raise ValueError("--limit must be between 1 and 5")
    if args.sleep_between_questions < 0:
        raise ValueError("--sleep-between-questions must be nonnegative")
    if not args.corpus.joinpath("manifest.csv").is_file():
        raise FileNotFoundError(args.corpus / "manifest.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    trace_dir = args.output_dir / "traces" / args.run_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "created_at": utc_now(),
                "configuration": "heterogeneous_static",
                "dataset": "LitQA2 public eval, five open-full-text questions",
                "question_ids": [q["question_id"] for q in questions],
                "corpus": str(args.corpus),
                "llm_model": args.model,
                "jev_model": "jev-1.13.0",
                "jev_input_price_usd_per_million": 0.042,
                "max_timesteps": args.max_timesteps,
                "timeout_seconds": args.timeout,
                "sleep_between_questions_seconds": args.sleep_between_questions,
                "grading": "first answer letter matched to prepared answer key",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    target = ResolvedModelTarget(
        target_name=args.model.replace("/", "-"),
        provider_name=provider,
        adapter_name="litellm",
        requested_model=args.model.split("/", 1)[1],
        litellm_model=args.model,
        temperature=0.0,
        litellm_params={"model": args.model},
        router_kwargs={},
        api_key_env=key_name,
    )
    bundle = ModelBundle.homogeneous(target)
    index_settings = settings_for(args.corpus, args.model, args.max_timesteps, args.timeout)
    await get_directory_index(index_settings, build=True)
    llm_logger = BaselineLLMLogger()
    prior_callbacks = list(getattr(litellm, "callbacks", []) or [])
    litellm.callbacks = [*prior_callbacks, llm_logger]
    completed = {
        row["question_id"]
        for line in results_path.read_text().splitlines()
        if (row := json.loads(line)) and row.get("run_id") == args.run_id
    } if results_path.exists() else set()
    identity = RunIdentity(args.run_id, "heterogeneous_static", "pilot", "LitQA2Pilot", "eval")
    try:
        remaining = [
            question for question in questions[: args.limit]
            if question["question_id"] not in completed
        ]
        for position, question in enumerate(remaining):
            question_id = question["question_id"]
            settings = settings_for(args.corpus, args.model, args.max_timesteps, args.timeout)
            settings.agent.rebuild_index = False
            record = await run_one(
                query=prompt_for(question),
                question_id=question_id,
                settings=settings,
                bundle=bundle,
                identity=identity,
                llm_logger=llm_logger,
                results_path=results_path,
                trace_path=trace_dir / f"{question_id}.trace.jsonl",
                pilot_gold_choice=answers[question_id],
            )
            print(
                f"Jev {question_id[:8]}: {record['status']}; "
                f"prediction={record.get('predicted_choice')}; "
                f"correct={record.get('correct')}; "
                f"wall_clock_ms={record['wall_clock_ms']}",
                flush=True,
            )
            if position + 1 < len(remaining) and args.sleep_between_questions:
                print(
                    f"Sleeping {args.sleep_between_questions:g}s before next question",
                    flush=True,
                )
                await asyncio.sleep(args.sleep_between_questions)
    finally:
        litellm.callbacks = prior_callbacks


if __name__ == "__main__":
    asyncio.run(main())
