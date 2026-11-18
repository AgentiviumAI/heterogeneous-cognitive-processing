from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv
from paperqa.agents.search import get_directory_index

from .config import ModelCatalog
from .executor import RunIdentity, failure_record, run_one
from .litqa2 import load_cases, validate_case_sources
from .paperqa_settings import ModelBundle, PaperQARunOptions, build_settings
from .providers import install_adaptive_rate_limit_recovery
from .tracing import BaselineLLMLogger
from .utils import RedactedStream, dump_jsonable, package_version, stable_user_question_id, utc_now

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "research" / "configs" / "model_targets.toml"
DEFAULT_SPLITS = REPO_ROOT / "docs" / "2024-10-16_litqa2-splits.json5"


def _add_common(parser: argparse.ArgumentParser) -> None:
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target")
    target.add_argument("--provider")
    parser.add_argument("--model", help="Required with --provider")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--paper-dir", required=True)
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--embedding", default="st-multi-qa-MiniLM-L6-cos-v1")
    parser.add_argument("--search-count", type=int, default=8)
    parser.add_argument("--agent-evidence-n", type=int, default=1)
    parser.add_argument("--max-timesteps", type=int)
    parser.add_argument("--timeout", type=float, default=500.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--index-concurrency", type=int, default=1)
    parser.add_argument("--index-batch-size", type=int, default=1)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--results", default="results/baseline.jsonl")
    parser.add_argument("--trace-root", default="results/traces")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--metadata-mode",
        choices=["upstream", "local"],
        default="upstream",
        help="upstream queries external metadata providers; local uses PDF metadata only",
    )

    parser.add_argument(
        "--multimodal-mode",
        choices=["enrich", "parse-only", "off"],
        default="enrich",
        help="enrich uses an enrichment LLM; parse-only parses media without LLM enrichment",
    )

    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=1,
    )
    parser.add_argument("--decision-backend", choices=["llm", "jev"], default="llm")
    parser.add_argument("--evidence-relevance-backend", choices=["llm", "jev"], default="llm")


def _resolve_target(args: argparse.Namespace, catalog: ModelCatalog):
    if args.target:
        return catalog.resolve_target(args.target)
    if not args.model:
        raise SystemExit("--model is required when using --provider")
    return catalog.resolve_inline(
        provider_name=args.provider,
        model=args.model,
        temperature=args.temperature,
    )


def _completed_ids(results_path: Path, run_id: str) -> set[str]:
    if not results_path.exists():
        return set()
    done: set[str] = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("run_id") == run_id and row.get("question_id"):
            done.add(str(row["question_id"]))
    return done


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dump_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in rows if r.get("correct") is not None]
    successful = [r for r in rows if r.get("status") != "fail"]
    return {
        "n_records": len(rows),
        "n_successful": len(successful),
        "n_failed": len(rows) - len(successful),
        "n_graded": len(graded),
        "accuracy": (sum(bool(r.get("correct")) for r in graded) / len(graded)) if graded else None,
        "mean_wall_clock_ms": (
            sum(float(r.get("wall_clock_ms") or 0.0) for r in rows) / len(rows)
            if rows else None
        ),
        "total_llm_calls": sum(int(r.get("llm_calls") or 0) for r in rows),
        "total_jev_calls": sum(int(r.get("jev_calls") or 0) for r in rows),
        "total_jev_cost_usd": sum(float(r.get("jev_cost_usd") or 0.0) for r in rows),
        "jev_cost_complete": all(bool(r.get("jev_cost_complete", True)) for r in rows),
        "total_input_tokens": sum(int(r.get("total_input_tokens") or 0) for r in rows),
        "total_output_tokens": sum(int(r.get("total_output_tokens") or 0) for r in rows),
        "total_cost_usd": (
            sum(float(r["total_cost_usd"]) for r in rows if r.get("total_cost_usd") is not None)
            if rows and all(r.get("total_cost_usd") is not None for r in rows)
            else None
        ),
    }


async def _run(args: argparse.Namespace) -> None:
    catalog = ModelCatalog.load(args.catalog)
    target = _resolve_target(args, catalog)
    restore_rate_limit_patch = (
        install_adaptive_rate_limit_recovery(
            max_attempts=target.rate_limit_recovery_attempts,
            default_wait_seconds=target.rate_limit_default_wait_seconds,
            max_wait_seconds=target.rate_limit_max_wait_seconds,
        )
        if target.adaptive_rate_limit_recovery
        else (lambda: None)
    )
    bundle = ModelBundle.homogeneous(target)
    options = PaperQARunOptions(
        paper_dir=args.paper_dir,
        index_name=args.index_name,
        manifest=args.manifest,
        embedding=args.embedding,
        search_count=args.search_count,
        agent_evidence_n=args.agent_evidence_n,
        max_timesteps=args.max_timesteps,
        timeout=args.timeout,
        retries=args.retries,
        index_concurrency=args.index_concurrency,
        index_batch_size=args.index_batch_size,
        rebuild_index=args.rebuild_index,
        verbosity=args.verbosity,
        metadata_mode=args.metadata_mode,
        multimodal_mode=args.multimodal_mode,
        max_concurrent_requests=args.max_concurrent_requests,
    )
    settings = build_settings(bundle, options)
    settings.agent.decision_backend = args.decision_backend
    settings.answer.evidence_relevance_backend = args.evidence_relevance_backend
    jev_enabled = args.decision_backend == "jev" or args.evidence_relevance_backend == "jev"
    if jev_enabled and not os.getenv("TYPESAFE_API_KEY"):
        raise SystemExit("TYPESAFE_API_KEY is required when a Jev backend is enabled")
    configuration = "heterogeneous_static" if jev_enabled else "homogeneous_llm"

    run_id = args.run_id or datetime.now().strftime(
        ("heterogeneous" if jev_enabled else "baseline") + "_%Y%m%d_%H%M%S"
    )
    results_path = Path(args.results)
    trace_dir = Path(args.trace_root) / run_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(results_path, run_id) if args.resume else set()

    run_config = {
        "run_id": run_id,
        "created_at": utc_now(),
        "mode": args.command,
        "models": bundle.public_metadata(),
        "paperqa": package_version("paper-qa"),
        "fhaviary": package_version("fhaviary"),
        "aviary_labbench": package_version("aviary.labbench"),
        "litellm": package_version("litellm"),
        "arguments": vars(args),
        "jev": {
            "decision_backend": settings.agent.decision_backend,
            "decision_model": settings.agent.jev_decision_model,
            "decision_input_price_usd_per_million": settings.agent.jev_input_price_usd_per_million,
            "evidence_relevance_backend": settings.answer.evidence_relevance_backend,
            "relevance_model": settings.answer.jev_relevance_model,
            "relevance_input_price_usd_per_million": (
                settings.answer.jev_input_price_usd_per_million
            ),
        },
    }
    _write_json(trace_dir / "run_config.json", run_config)

    llm_logger = BaselineLLMLogger()
    previous_callbacks = list(getattr(litellm, "callbacks", []) or [])
    litellm.callbacks = [*previous_callbacks, llm_logger]
    rows: list[dict[str, Any]] = []

    try:
        if args.command == "user":
            question_id = stable_user_question_id(args.question)
            if question_id not in completed:
                identity = RunIdentity(run_id, configuration, "user", None, None)
                rows.append(
                    await run_one(
                        query=args.question,
                        question_id=question_id,
                        settings=settings,
                        bundle=bundle,
                        identity=identity,
                        llm_logger=llm_logger,
                        results_path=results_path,
                        trace_path=trace_dir / f"{question_id}.trace.jsonl",
                    )
                )
        else:
            cases = load_cases(
                settings=settings,
                split=args.split,
                splits_file=args.splits_file,
                limit=None if args.limit == "all" else int(args.limit),
                seed=args.seed,
                question_id=args.question_id,
            )
            identity = RunIdentity(run_id, configuration, "litqa", "LitQA2", args.split)
            index = None
            if not args.skip_source_validation:
                index = await get_directory_index(settings=settings, build=True)

            for case in cases:
                if case.question_id in completed:
                    continue
                trace_path = trace_dir / f"{case.question_id}.trace.jsonl"
                if index is not None:
                    try:
                        await validate_case_sources(case, settings, index)
                    except Exception as exc:
                        rows.append(
                            failure_record(
                                question_id=case.question_id,
                                question=getattr(case.query, "question", str(case.query)),
                                identity=identity,
                                settings=settings,
                                bundle=bundle,
                                results_path=results_path,
                                trace_path=trace_path,
                                error=f"source_validation: {type(exc).__name__}: {exc}",
                            )
                        )
                        continue
                rows.append(
                    await run_one(
                        query=case.query,
                        question_id=case.question_id,
                        settings=settings,
                        bundle=bundle,
                        identity=identity,
                        llm_logger=llm_logger,
                        results_path=results_path,
                        trace_path=trace_path,
                        sources=case.sources,
                    )
                )
    finally:
        litellm.callbacks = previous_callbacks
        restore_rate_limit_patch()

    summary = _summary(rows)
    summary.update({"run_id": run_id, "model": target.public_metadata(), "timestamp": utc_now()})
    _write_json(trace_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provider-agnostic PaperQA baseline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    litqa = sub.add_parser("litqa", help="Run LitQA2 benchmark questions")
    _add_common(litqa)
    litqa.add_argument("--split", choices=["train", "eval", "test"], default="eval")
    litqa.add_argument("--splits-file", default=str(DEFAULT_SPLITS))
    litqa.add_argument("--limit", default="1", help="Integer count or 'all'")
    litqa.add_argument("--seed", type=int, default=20261003)
    litqa.add_argument("--question-id")
    litqa.add_argument("--skip-source-validation", action="store_true")

    user = sub.add_parser("user", help="Run an ordinary user question")
    _add_common(user)
    user.add_argument("--question", required=True)
    return parser


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    original_stderr = sys.stderr
    sys.stderr = RedactedStream(original_stderr)
    for configured_logger in [logging.getLogger(), *logging.root.manager.loggerDict.values()]:
        if isinstance(configured_logger, logging.Logger):
            for handler in configured_logger.handlers:
                if (
                    isinstance(handler, logging.StreamHandler)
                    and handler.stream is original_stderr
                ):
                    handler.setStream(sys.stderr)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    asyncio.run(_run(args))
