#!/usr/bin/env python3
"""Run the same PaperQA experiment over multiple named model targets.

Example:
  python research/scripts/run_llm_matrix.py \
    --targets opencode-default groq-default \
    -- litqa --paper-dir /DATA/papers --split eval --limit 10

Everything after `--` is forwarded to run_llm_baseline.py. The matrix runner only
injects --target and a target-specific --run-id.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("forward", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    forwarded = list(args.forward)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded or forwarded[0] not in {"litqa", "user"}:
        raise SystemExit("Forwarded args must start with `litqa` or `user` after `--`.")

    runner = Path(__file__).with_name("run_llm_baseline.py")
    prefix = args.run_prefix or datetime.now().strftime("matrix_%Y%m%d_%H%M%S")
    failures: list[str] = []

    for target in args.targets:
        run_id = f"{prefix}_{safe(target)}"
        command = [
            sys.executable,
            str(runner),
            forwarded[0],
            "--target",
            target,
            "--run-id",
            run_id,
            *forwarded[1:],
        ]
        print("\n===", target, "===")
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.append(target)
            if not args.continue_on_error:
                raise SystemExit(result.returncode)

    if failures:
        raise SystemExit(f"Failed targets: {', '.join(failures)}")


if __name__ == "__main__":
    main()
