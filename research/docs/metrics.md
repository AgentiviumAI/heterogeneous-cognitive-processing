# Metrics and Result Schema

All experiment runners should emit JSONL using the same field names.

## Required fields

```json
{
  "run_id": "string",
  "question_id": "string",
  "configuration": "homogeneous_llm | heterogeneous_static | decision_micro",
  "operation": "string",
  "cognitive_mode": "decision | generation | mixed | unknown",
  "backend": "llm | jev | clef",
  "model": "string",
  "latency_ms": 0.0,
  "input_tokens": 0,
  "output_tokens": 0,
  "cost_usd": 0.0,
  "correct": null,
  "score": null,
  "answer": null,
  "metadata": {}
}
```

## End-to-end metrics

Required:
- LitQA2 accuracy / task score,
- total wall-clock latency,
- total model/API cost,
- number of model calls.

Recommended:
- latency per question,
- decision/generation call counts,
- failure/retry counts.

## Decision-step metrics

Required:
- operation-level correctness or agreement with upstream target where available,
- latency,
- cost.

If a gold decision label does not exist, document the proxy/evaluation rule explicitly instead of silently inventing one.

## Measurement inventory task

For each metric, record:
- whether PaperQA already exposes it,
- exact hook/callback location,
- whether extra instrumentation is required,
- how values are aggregated.
