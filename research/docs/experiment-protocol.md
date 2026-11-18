# Experiment Protocol

## Phase 1 — Smoke test

Run 5–10 LitQA2 questions to verify:
1. PaperQA baseline executes end-to-end.
2. Trajectories/model calls can be logged.
3. At least one decision-oriented step can be replayed through Jev.

Do not scale the benchmark before these three checks pass.

## Phase 2 — Homogeneous baseline

Run the shared pilot subset with:
- decision-oriented steps -> LLM
- generative steps -> LLM

Collect end-to-end and per-call metrics using the shared schema.

## Phase 3 — Decision affinity micro-evaluation

Select 1–2 clear decision-oriented operations first:
- tool/action selection,
- evidence relevance / evidence selection.

Replay the same inputs through:
- LLM baseline,
- Jev,
- Clef (optional).

Measure quality, latency, and cost.

## Phase 4 — Static heterogeneous execution

Run the same LitQA2 subset with:
- selected decision steps -> Jev,
- generative steps -> baseline LLM.

Compare against the homogeneous baseline.

## Minimum arXiv evidence

The paper is considered experimentally supported if we can show:
1. PaperQA trajectories contain both decision and generation modes.
2. Decision steps show measurable Jev/LLM affinity differences.
3. Static heterogeneous execution changes the end-to-end quality-latency-cost trade-off.

## Out of scope for this phase

- learned router/scheduler,
- online adaptation,
- load-aware placement,
- automatic cognitive-mode discovery,
- multi-workload generalization beyond optional validation.
