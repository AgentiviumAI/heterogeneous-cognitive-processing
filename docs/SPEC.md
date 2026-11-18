# SPEC — Heterogeneous Cognitive Processing for PaperQA

## 1. Objective

Implement a heterogeneous PaperQA execution path where bounded decision-oriented processing is handled by Jev and open-ended generation remains handled by the existing LLM.

The goal is to compare this heterogeneous execution against the reproduced original PaperQA baseline using end-to-end LitQA2 metrics.

## 2. Core Constraint

**Only decompose internal cognitive processing. Do not change PaperQA's external input or output contracts.**

Principle:

> Same input contract, same output contract, different internal cognitive backend.

This means:

- Keep the same LitQA2 questions, corpus/index, relevant settings, tool definitions, and workflow inputs.
- Keep the same downstream-compatible PaperQA outputs and object structures.
- Only replace how selected internal outputs are produced.
- The rest of PaperQA should not need to know whether a decision was produced by the LLM or Jev.

Example:

```text
Original:
(question, evidence text)
→ LLM
→ Context(summary, relevance_score)

Heterogeneous:
(question, evidence text)
├─ LLM → summary
└─ Jev → relevance_score
→ Context(summary, relevance_score)
```

---

## 3. Decomposition Targets

### 3.1 Evidence Relevance

Current behavior:

```text
question + evidence chunk
        ↓
       LLM
        ↓
summary + relevance_score
```

Heterogeneous behavior:

```text
question + evidence chunk
        ├─ LLM → summary
        └─ Jev → relevance_score
                     ↓
          Context(summary, score)
```

Responsibilities:

- LLM: generate the evidence summary.
- Jev: produce the bounded relevance score.
- Output contract remains the same `Context` object expected by PaperQA.

This is the first and highest-priority decomposition target.

---

### 3.2 Tool / Action Selection

Current behavior:

```text
agent state + available tools
          ↓
         LLM
          ↓
selected tool + tool arguments
```

Heterogeneous behavior:

```text
agent state + available tools
          ↓
         Jev
          ↓
      selected tool
          ↓
LLM generates arguments only when needed
          ↓
same ToolRequestMessage as original PaperQA
```

Candidate tools include:

- `paper_search`
- `gather_evidence`
- `gen_answer`
- `reset`
- `complete`

Examples:

- `paper_search`
  - Jev selects the tool.
  - LLM generates search query / year arguments.

- `gather_evidence`
  - Jev selects the tool.
  - LLM generates or refines an argument only if the existing PaperQA interface requires it.

- `gen_answer`
  - Jev selects the tool.
  - Final answer generation remains LLM-based.

- `reset`
  - Jev selects the tool.
  - No open-ended generation is required.

- `complete`
  - Jev selects the termination action.
  - Completion status can also be handled as a bounded decision.

Output must remain compatible with the original PaperQA `ToolRequestMessage` / tool interface.

---

### 3.3 Complete / Stop Decision

Current behavior:

```text
current state
    ↓
   LLM
    ↓
complete(has_successful_answer=...)
```

Heterogeneous behavior:

```text
current state
    ↓
   Jev
    ↓
CONTINUE
COMPLETE_SUCCESS
COMPLETE_UNSURE
```

Mapping:

```text
CONTINUE
→ continue the agent workflow

COMPLETE_SUCCESS
→ complete(has_successful_answer=True)

COMPLETE_UNSURE
→ complete(has_successful_answer=False)
```

The downstream PaperQA completion interface must remain unchanged.

---

## 4. Components That Remain LLM / Existing Logic

Do not replace the following with Jev:

- Search-query generation
- Evidence-summary generation
- Final-answer generation
- Retrieval / embedding search
- MMR retrieval
- Deterministic relevance sorting/filtering
- Citation formatting
- Other deterministic workflow logic

Jev is used only for bounded decision-oriented operations.

---

## 5. Evaluation Strategy

### 5.1 Primary Evaluation: End-to-End

The main comparison is:

```text
Original PaperQA baseline
LLM handles the original workflow

vs.

Heterogeneous PaperQA
Jev handles selected decisions
LLM handles generation
```

Use the same LitQA2 subset, corpus/index, model configuration where applicable, and experimental conditions.

Primary metrics:

- LitQA2 final-answer accuracy
- Total wall-clock latency
- Total cost
- Number of model/API calls
- Token usage where available
- Workflow / tool-step count

No intermediate gold labels are required.

---

### 5.2 Micro-Evaluation

Micro-evaluation is only for analysis and interpretation of Jev vs LLM behavior.

Use the same intermediate input/state from the baseline trajectory and replay it through Jev.

#### Evidence relevance

```text
same question + evidence chunk
├─ baseline LLM → relevance score
└─ Jev          → relevance score
```

Analyze:

- score difference / agreement
- Jev confidence / probability distribution
- latency
- token usage
- cost

#### Tool selection

```text
same agent state + available tools
├─ baseline LLM → selected tool
└─ Jev          → selected tool
```

Analyze:

- agreement / disagreement
- selected action
- Jev confidence / probabilities
- latency
- token usage
- cost

#### Termination

```text
same relevant agent state
├─ baseline LLM → termination behavior
└─ Jev          → termination decision
```

Analyze:

- agreement / disagreement
- decision
- Jev confidence / probabilities
- latency
- token usage
- cost

Important:

- Do not treat the baseline LLM output as gold.
- Do not require manually annotated intermediate labels.
- End-to-end LitQA2 correctness remains the main quality measure.

---

## 6. Existing PaperQA Logging

PaperQA already contains or exposes several useful fields:

- final answer
- raw answer
- optional answer reasoning
- success / unsure status
- evidence summaries
- evidence relevance scores
- contexts used by the final answer
- total LLM cost
- token counts aggregated by model
- tool-name history
- configuration hash
- citations / references
- agent status

However, this is not sufficient for a clean heterogeneous comparison.

---

## 7. Required Additional Logging

### 7.1 Per-Question End-to-End Record

Each benchmark question should produce one summary record containing at least:

```json
{
  "run_id": "...",
  "question_id": "...",
  "configuration": "original_paperqa | heterogeneous_static",

  "final_answer": "...",
  "predicted_choice": "...",
  "gold_choice": "...",
  "correct": true,

  "status": "success",

  "wall_clock_ms": 0.0,

  "total_cost_usd": 0.0,
  "llm_cost_usd": 0.0,
  "jev_cost_usd": 0.0,

  "llm_calls": 0,
  "jev_calls": 0,

  "token_counts": {},

  "tool_steps": 0,

  "config_md5": "...",
  "commit_sha": "...",
  "trace_file": "..."
}
```

Required additions compared with the current PaperQA session data:

1. LitQA2 `question_id`
2. Final correctness
3. Actual end-to-end wall-clock latency
4. Separate LLM and Jev cost
5. Number of LLM calls
6. Number of Jev calls
7. Full trace reference
8. Commit SHA / reproducibility metadata

---

### 7.2 Per-Operation Trace

Log every relevant model/API operation.

Recommended schema:

```json
{
  "run_id": "...",
  "question_id": "...",
  "step": 0,

  "operation": "evidence_relevance",
  "cognitive_mode": "decision",
  "backend": "jev",
  "model": "...",

  "input": {},
  "output": {},

  "latency_ms": 0.0,

  "input_tokens": null,
  "output_tokens": null,

  "cost_usd": 0.0,

  "confidence": null,
  "probabilities": null,

  "retry_count": 0,
  "error": null
}
```

Operations should distinguish at least:

- `tool_selection`
- `tool_argument_generation`
- `evidence_summary`
- `evidence_relevance`
- `final_answer`
- `termination`

---

### 7.3 Full Tool Trajectory

The current `tool_history` stores only tool names.

Add:

- selected tool
- tool arguments
- backend responsible for the decision
- associated step number
- relevant output / status

Example:

```json
{
  "step": 3,
  "operation": "tool_selection",
  "backend": "jev",
  "selected_tool": "paper_search",
  "arguments": {
    "query": "...",
    "min_year": null,
    "max_year": null
  }
}
```

The final downstream tool request must still match the original PaperQA interface.

---

## 8. Jev-Specific Logging

For every Jev request, log all fields that the API exposes and all comparison-relevant client-side measurements.

Where available, retain:

- actual Jev model identifier
- decision type, e.g. Choice / Score / boolean-style decision
- selected choice or numerical score
- confidence
- probability distribution over choices / score levels
- input token count
- output token count

Additionally measure client-side:

- request latency
- retry count
- API errors
- timeout status

For cost:

- calculate Jev cost from reported usage and the pricing configuration used for that experiment
- record the pricing/version used in experiment metadata
- do not silently omit Jev cost from heterogeneous totals

If a Jev response does not expose a field, record it as `null` rather than estimating it without documentation.

---

## 9. Important Logging Detail: Preserve Intermediate Inputs

The normal PaperQA response filtering can remove the original evidence text from returned `Context` objects.

Therefore, log intermediate decision inputs before user-facing response filtering.

For evidence relevance, preserve at least:

```text
question
evidence text/chunk
document/chunk identifier
summary
relevance score
backend
```

This is required for reproducible micro-evaluation and debugging.

---

## 10. Experimental Flow

```text
LitQA2 shared subset
        │
        ├──────────────────────────┐
        │                          │
        ▼                          ▼
Original PaperQA             Heterogeneous PaperQA
baseline reproduction        Jev decisions + LLM generation
        │                          │
        │                          │
        └────────────┬─────────────┘
                     ▼
             End-to-end comparison
          accuracy / latency / cost
                     │
                     ▼
             Micro-analysis
      replay same decision inputs through
             baseline LLM vs Jev
```

Recommended implementation order:

1. Ensure the shared PaperQA / LitQA2 pilot runs successfully.
2. Add the required logging and trace schema.
3. Capture baseline intermediate decision inputs.
4. Implement Jev client / adapter.
5. Decompose evidence relevance.
6. Run heterogeneous PaperQA end-to-end.
7. Add tool-selection decomposition.
8. Add termination decomposition.
9. Perform end-to-end comparison.
10. Use replayed decision states for micro-analysis.

---

## 11. Scope Summary

The heterogeneous system should follow one rule:

> **Jev handles bounded decisions; the LLM handles open-ended generation, while PaperQA's external behavior and interfaces remain unchanged.**

Three decomposition targets:

1. Evidence relevance
2. Tool/action selection
3. Complete/stop decision

Evaluation:

- Main: end-to-end LitQA2 accuracy, latency, cost, and execution statistics.
- Secondary: micro-analysis of Jev vs baseline LLM on identical intermediate inputs.
- No intermediate gold labels are required.

Instrumentation:

- Extend the current PaperQA logging with complete end-to-end timing/cost accounting, full operation traces, full tool calls, Jev metadata, backend-specific usage, and reproducibility metadata.
