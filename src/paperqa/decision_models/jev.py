"""Jev relevance scoring for the heterogeneous PaperQA experiment."""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_QUESTION_ID = "evidence_relevance"

# Score supports no more than ten levels. Six descriptive levels map to
# PaperQA's integer 0-10 score while preserving Jev's continuous value.
RELEVANCE_LEVELS = (
    "No information useful for answering the question",
    "Only a remote topical connection; no usable answer evidence",
    "Some related facts, but little direct answer evidence",
    "Partly answers the question with specific relevant facts",
    "Strong, specific evidence for most of the question",
    "Direct and sufficient evidence for answering the question",
)


@dataclass(frozen=True)
class JevRelevanceResult:
    score: int
    raw_score: float
    confidence: float
    probabilities: dict[str, float]
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    retry_count: int


def build_relevance_request(question: str, evidence_text: str, model: str) -> dict[str, Any]:
    if not question.strip() or not evidence_text.strip():
        raise ValueError("question and evidence text must be non-empty")
    return {
        "model": model,
        "state": {"question": question, "evidence_text": evidence_text},
        "questions": {
            JEV_QUESTION_ID: {
                "type": "score",
                "instructions": (
                    "How useful is the evidence_text for answering the question? "
                    "Rate relevance of the evidence, not writing quality."
                ),
                "criteria": list(RELEVANCE_LEVELS),
            }
        },
    }


def parse_relevance_response(
    body: dict[str, Any], latency_ms: float, retry_count: int, price: float
) -> JevRelevanceResult:
    try:
        answer = body["answers"][JEV_QUESTION_ID]
        usage = body["usage"]
        raw_score = float(answer["score"])
        confidence = float(answer["confidence"])
        probabilities = {
            str(i): float(answer["probabilities"][str(i)]) for i in range(len(RELEVANCE_LEVELS))
        }
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        model = body["model"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid Jev relevance response") from exc
    invalid_answer = (
        answer.get("type") != "score"
        or not isinstance(model, str)
        or not 0 <= raw_score <= len(RELEVANCE_LEVELS) - 1
        or not 0 <= confidence <= 1
    )
    invalid_usage = (
        not isinstance(input_tokens, int)
        or not isinstance(output_tokens, int)
        or input_tokens < 0
        or output_tokens < 0
    )
    invalid_probabilities = not all(
        math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()
    ) or not math.isclose(sum(probabilities.values()), 1, abs_tol=0.02)
    if invalid_answer or invalid_usage or invalid_probabilities:
        raise ValueError("Invalid Jev relevance response")
    return JevRelevanceResult(
        score=min(10, max(0, math.floor(raw_score * 2 + 0.5))),
        raw_score=raw_score,
        confidence=confidence,
        probabilities=probabilities,
        model=model,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=input_tokens * price / 1_000_000,
        retry_count=retry_count,
    )


class JevRelevanceClient:
    def __init__(
        self,
        *,
        model: str,
        input_price_usd_per_million: float,
        api_key: str | None = None,
        timeout_s: float = 30,
        retries: int = 2,
    ) -> None:
        self.model = model
        self.price = input_price_usd_per_million
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        self.timeout_s = timeout_s
        self.retries = retries
        if not self.api_key:
            raise ValueError("TYPESAFE_API_KEY is required for Jev relevance scoring")
        if self.price < 0:
            raise ValueError("Jev input price must be non-negative")
        self._http = httpx.AsyncClient(timeout=self.timeout_s)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def score(self, question: str, evidence_text: str) -> JevRelevanceResult:
        payload = build_relevance_request(question, evidence_text, self.model)
        start = perf_counter()
        for attempt in range(self.retries + 1):
            try:
                response = await self._http.post(
                    JEV_ENDPOINT,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise TypeError("Invalid Jev JSON response")
                return parse_relevance_response(
                    body, (perf_counter() - start) * 1000, attempt, self.price
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {429, 529} or attempt == self.retries:
                    raise
            except httpx.RequestError:
                if attempt == self.retries:
                    raise
            await asyncio.sleep(0.5 * (2**attempt))
        raise AssertionError("unreachable")
