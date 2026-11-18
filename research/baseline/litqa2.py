from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from paperqa.docs import Docs

try:
    from aviary.envs.labbench import (
        GradablePaperQAEnvironment,
        LABBenchDatasets,
        TextQATaskDataset,
    )
except ImportError:  # optional dataset extra
    GradablePaperQAEnvironment = None  # type: ignore[assignment]
    LABBenchDatasets = None  # type: ignore[assignment]
    TextQATaskDataset = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LitQA2Case:
    query: Any
    sources: Any

    @property
    def question_id(self) -> str:
        return str(self.query.question_id)


def _split_section(text: str, split: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(split)}\s*:\s*\{{", text)
    if not match:
        raise ValueError(f"Split {split!r} not found in reproduction file")
    start = text.find("{", match.start())
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    raise ValueError(f"Unclosed split section {split!r}")


def split_values(path: str | Path, split: str, field: str) -> list[str]:
    section = _split_section(Path(path).read_text(encoding="utf-8"), split)
    match = re.search(rf"(?s)\b{re.escape(field)}\s*:\s*\[(.*?)\]", section)
    if not match:
        raise ValueError(f"Field {field!r} not found for split {split!r}")
    return re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', match.group(1))


def load_cases(
    *,
    settings: Any,
    split: str,
    splits_file: str | Path,
    limit: int | None,
    seed: int,
    question_id: str | None = None,
) -> list[LitQA2Case]:
    if TextQATaskDataset is None or LABBenchDatasets is None:
        raise RuntimeError(
            "LitQA2 dataset support is unavailable. Install the labbench datasets "
            "extra matching the project's pinned Aviary version."
        )

    wanted_ids = split_values(splits_file, split, "question_ids")
    wanted = set(wanted_ids)

    # PaperQA's historical train/eval partition is carved out of LAB-Bench train.
    hf_split = "test" if split == "test" else "train"
    dataset = TextQATaskDataset(
        settings=settings,
        dataset=LABBenchDatasets.LIT_QA2,
        split=hf_split,
        read_data_kwargs={"randomize": False},
    )

    rows = dataset.data[dataset.data["id"].astype(str).isin(wanted)].copy()
    found = set(rows["id"].astype(str).tolist())
    missing = wanted - found
    if missing:
        example = next(iter(missing))
        raise RuntimeError(
            f"LitQA2 dataset is missing {len(missing)} IDs from PaperQA's {split} "
            f"reproduction split; example: {example}"
        )

    order = {qid: i for i, qid in enumerate(wanted_ids)}
    rows["__order"] = rows["id"].astype(str).map(order)
    dataset.data = (
        rows.sort_values("__order").drop(columns=["__order"]).reset_index(drop=True)
    )

    if question_id:
        indexes = [
            i
            for i, qid in enumerate(dataset.data["id"].astype(str))
            if qid == question_id
        ]
        if not indexes:
            raise ValueError(f"Question {question_id} is not in PaperQA's {split} split")
    else:
        indexes = list(range(len(dataset)))
        if limit is not None and limit < len(indexes):
            rng = random.Random(seed)
            indexes = sorted(rng.sample(indexes, limit))

    return [
        LitQA2Case(query=dataset._make_query(i), sources=dataset._make_sources(i))
        for i in indexes
    ]


async def validate_case_sources(case: LitQA2Case, settings: Any, index: Any) -> None:
    if GradablePaperQAEnvironment is None:
        raise RuntimeError("GradablePaperQAEnvironment is unavailable")
    env = GradablePaperQAEnvironment(
        query=case.query,
        settings=settings,
        docs=Docs(),
        sources=case.sources,
        session_id=UUID(case.question_id),
    )
    await env.validate_sources(index)


def environment_class() -> Any:
    if GradablePaperQAEnvironment is None:
        raise RuntimeError("GradablePaperQAEnvironment is unavailable")
    return GradablePaperQAEnvironment
