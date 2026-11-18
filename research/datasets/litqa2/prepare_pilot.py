"""Create reproducible, blinded LitQA2 pilot questions from the public file."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
from operator import itemgetter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).with_name("litqa-v2-public.jsonl")
PARQUET = Path(__file__).with_name("train-00000-of-00001.parquet")
OPEN_FULLTEXT = Path(__file__).with_name("europe-pmc-open-fulltext.json")
SPLITS = ROOT / "docs/2024-10-16_litqa2-splits.json5"
PILOT_CONFIG = ROOT / "research/configs/pilot.yaml"


def eval_ids() -> set[str]:
    """Read the eval question ID array from the repository's pinned split file."""
    split = SPLITS.read_text(encoding="utf-8").split("  eval: {", 1)[1]
    array = split.split("question_ids: [", 1)[1].split("]", 1)[0]
    return set(re.findall(r'"([0-9a-f-]{36})"', array))


def prepare_pilot(
    count: int, seed: int, eligible_ids: set[str] | None = None
) -> tuple[list[dict], list[dict]]:
    """Sample eval questions and shuffle choices without exposing the answer."""
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]
    allowed = eval_ids()
    if eligible_ids is not None:
        if not eligible_ids <= allowed:
            raise ValueError("Open full-text question IDs must belong to the eval split")
        allowed &= eligible_ids
    candidates = sorted((row for row in rows if row["id"] in allowed), key=itemgetter("id"))
    if len(candidates) != len(allowed):
        raise ValueError("Public LitQA2 data does not match the pinned eval split")
    if not 1 <= count <= len(candidates):
        raise ValueError(f"Count must be between 1 and {len(candidates)}")

    rng = random.Random(seed)
    selected = rng.sample(candidates, count)
    questions, answers = [], []
    for row in selected:
        choices = [(row["ideal"], True), *((choice, False) for choice in row["distractors"])]
        rng.shuffle(choices)
        questions.append({
            "question_id": row["id"],
            "question": row["question"],
            "choices": [choice for choice, _ in choices],
        })
        answers.append({
            "question_id": row["id"],
            "gold_choice": string.ascii_uppercase[
                next(index for index, (_, is_gold) in enumerate(choices) if is_gold)
            ],
        })
    return questions, answers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PILOT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["dataset"]
    if config["benchmark"] != "LitQA2" or config["split"] != "eval":
        raise ValueError("This pilot preparation supports only the LitQA2 eval split")
    selection = config.get("selection")
    if selection == "europe_pmc_open_fulltext":
        eligibility = json.loads(OPEN_FULLTEXT.read_text(encoding="utf-8"))
        eligible_ids = {record["id"] for record in eligibility["records"]}
    elif selection is None:
        eligible_ids = None
    else:
        raise ValueError(f"Unsupported pilot selection: {selection}")
    questions, answers = prepare_pilot(
        config["subset_size"], config["subset_seed"], eligible_ids
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("questions.jsonl", questions), ("answer_key.jsonl", answers)):
        path = args.output_dir / name
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        print(f"{path}: {len(records)} records")
    manifest = {
        "benchmark": config["benchmark"],
        "split": config["split"],
        "subset_size": config["subset_size"],
        "subset_seed": config["subset_seed"],
        "selection": selection,
        "question_ids": [record["question_id"] for record in questions],
        "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "source_parquet_sha256": hashlib.sha256(PARQUET.read_bytes()).hexdigest(),
        "source_revision": "5c77cec648430f30611808808861eb86f81d5eaa",
    }
    if eligible_ids is not None:
        manifest["eligibility_sha256"] = hashlib.sha256(OPEN_FULLTEXT.read_bytes()).hexdigest()
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
