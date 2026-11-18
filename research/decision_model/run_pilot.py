"""Run the five-question open-full-text LitQA2 integration pilot."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import string
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from paperqa.agents.main import run_agent
from paperqa.agents.search import get_directory_index
from paperqa.docs import Docs
from paperqa.settings import AgentSettings, AnswerSettings, IndexSettings, ParsingSettings, Settings


class RedactedStderr:
    """Keep provider error URLs from printing local API keys."""

    def __init__(self, stream, secrets: list[str]):
        self.stream = stream
        self.secrets = [secret for secret in secrets if secret]

    def write(self, value: str) -> int:
        sanitized = value
        for secret in self.secrets:
            sanitized = sanitized.replace(secret, "[REDACTED]")
        self.stream.write(sanitized)
        return len(value)

    def flush(self) -> None:
        self.stream.flush()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="groq/openai/gpt-oss-120b")
    parser.add_argument("--backends", nargs="+", choices=("llm", "jev"), default=("llm", "jev"))
    parser.add_argument("--question-limit", type=int, default=5)
    args = parser.parse_args()
    load_dotenv()
    original_stderr = sys.stderr
    redacted_stderr = RedactedStderr(
        original_stderr,
        [os.getenv(name, "") for name in ("GEMINI_API_KEY", "GROQ_API_KEY", "TYPESAFE_API_KEY")],
    )
    sys.stderr = redacted_stderr
    for logger in [logging.getLogger(), *logging.root.manager.loggerDict.values()]:
        if not isinstance(logger, logging.Logger):
            continue
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream is original_stderr:
                handler.setStream(redacted_stderr)
    provider_key = "GEMINI_API_KEY" if args.model.startswith("gemini/") else "GROQ_API_KEY"
    if not os.getenv(provider_key) or not os.getenv("TYPESAFE_API_KEY"):
        raise RuntimeError(f"{provider_key} and TYPESAFE_API_KEY are required")
    questions = [json.loads(line) for line in args.questions.read_text().splitlines()]
    answers = {row["question_id"]: row["gold_choice"] for line in args.answer_key.read_text().splitlines() if (row := json.loads(line))}
    if len(questions) != 5 or len(answers) != 5:
        raise ValueError("Expected exactly five questions and answers")
    if not 1 <= args.question_limit <= 5:
        raise ValueError("Question limit must be between one and five")
    model = args.model
    common = dict(
        llm=model,
        summary_llm=model,
        embedding="sparse",
        parsing=ParsingSettings(use_doc_details=False, multimodal=False),
        agent=AgentSettings(
            agent_llm=model,
            index=IndexSettings(
                name="litqa2-five-fulltext",
                paper_directory=args.corpus,
                manifest_file="manifest.csv",
                index_directory=args.corpus.parent / "hcp-litqa2-fulltext-index",
            ),
            max_timesteps=5,
            timeout=180,
            search_count=2,
        ),
    )
    index_settings = Settings(**common)
    await get_directory_index(index_settings, build=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        completed = {
            (row["backend"], row["question_id"])
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if (row := json.loads(line))
        }
    with args.output.open("a", encoding="utf-8") as output:
        for backend in args.backends:
            settings = Settings(**common)
            settings.agent.rebuild_index = False
            settings.agent.decision_backend = backend
            settings.answer = AnswerSettings(evidence_relevance_backend=backend, evidence_k=3)
            for question in questions[: args.question_limit]:
                if (backend, question["question_id"]) in completed:
                    continue
                letters = string.ascii_uppercase
                choices = "\n".join(
                    f"{letters[i]}. {choice}" for i, choice in enumerate(question["choices"])
                )
                prompt = (
                    f"{question['question']}\n\n{choices}\n\n"
                    "Choose the best answer. Begin your final answer with its letter, "
                    "followed by a short explanation and citation."
                )
                start = time.monotonic()
                try:
                    result = await run_agent(Docs(), prompt, settings)
                    raw = result.session.answer.strip()
                    match = re.search(r"(?:^|\n)\s*(?:Answer:\s*)?([A-Z])(?:[.):\s]|$)", raw, re.I)
                    prediction = match.group(1).upper() if match else None
                    record = {
                        "question_id": question["question_id"],
                        "backend": backend,
                        "model": model,
                        "status": str(result.status),
                        "answer": raw,
                        "prediction": prediction,
                        "correct": prediction == answers[question["question_id"]],
                        "seconds": round(time.monotonic() - start, 2),
                        "tracked_cost_usd": result.session.cost,
                        "token_counts": result.session.token_counts,
                        "tool_history": result.session.tool_history,
                    }
                except Exception as exc:
                    record = {
                        "question_id": question["question_id"],
                        "backend": backend,
                        "model": model,
                        "status": "exception",
                        "error": f"{type(exc).__name__}: {exc}",
                        "seconds": round(time.monotonic() - start, 2),
                    }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(f"{backend} {question['question_id'][:8]} {record['status']} {record.get('prediction')} {record['seconds']}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
