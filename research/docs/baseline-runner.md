# Provider-agnostic PaperQA baseline runner

## Goal

The baseline runner keeps the upstream PaperQA workflow unchanged while separating:

1. **Provider/model resolution** — endpoint, auth environment variable, LiteLLM routing.
2. **PaperQA settings** — the model roles and upstream `ToolSelector` configuration.
3. **Dataset adapter** — LitQA2 split loading and source validation.
4. **Execution** — one PaperQA episode at a time.
5. **Instrumentation** — end-to-end JSONL plus detailed per-question traces.

The provider layer must not contain benchmark logic, and the benchmark layer must not
contain provider-specific URLs or API-key names.

## Layout

```text
research/
├── baseline/
│   ├── config.py
│   ├── executor.py
│   ├── litqa2.py
│   ├── paperqa_settings.py
│   ├── tracing.py
│   └── providers/
│       ├── base.py
│       ├── litellm.py
│       └── registry.py
├── configs/
│   └── model_targets.toml
└── scripts/
    ├── run_llm_baseline.py
    └── run_llm_matrix.py
```

## Provider vs target

A **provider** describes transport/authentication. A **target** is one concrete model
run through a provider.

```toml
[providers.opencode]
adapter = "openai_compatible"
litellm_prefix = "openai"
api_base = "https://opencode.ai/zen/v1"
api_key_env = "OPENCODE_API_KEY"

[targets.opencode-model-a]
provider = "opencode"
model = "model-a"
temperature = 0.0
```

API keys are read from the environment and are never written into the catalog or run
metadata.

## Three standard checks

### 1. One LitQA2 case

```bash
python research/scripts/run_llm_baseline.py litqa \
  --provider opencode \
  --model "$OPENCODE_MODEL" \
  --paper-dir /DATA/paperqa/litqa2/eval \
  --index-name litqa2_eval \
  --split eval \
  --limit 1
```

This is the upstream pipeline smoke test. It should use the real `ToolSelector`, the
LitQA2 gradable environment, source validation, and the normal PaperQA tools.

### 2. Real user prompt

```bash
python research/scripts/run_llm_baseline.py user \
  --provider opencode \
  --model "$OPENCODE_MODEL" \
  --paper-dir /DATA/paperqa/litqa2/eval \
  --index-name litqa2_eval \
  --question "What are the main findings of these papers?"
```

This has no gold answer or accuracy. It verifies the same upstream workflow with an
ordinary user query.

### 3. Full LitQA2 evaluation split

```bash
python research/scripts/run_llm_baseline.py litqa \
  --provider opencode \
  --model "$OPENCODE_MODEL" \
  --paper-dir /DATA/paperqa/litqa2/eval \
  --index-name litqa2_eval \
  --split eval \
  --limit all \
  --run-id baseline_opencode_model_a_eval_v1
```

Use `--resume` with the same run ID after an interruption.

## Named targets

For repeated experiments, put the exact model ID in `research/configs/model_targets.toml`
and use:

```bash
python research/scripts/run_llm_baseline.py litqa \
  --target opencode-default \
  --paper-dir /DATA/paperqa/litqa2/eval \
  --index-name litqa2_eval \
  --split eval \
  --limit 10
```

The run metadata records the resolved provider/model configuration.

## Model matrix

The same experiment can be repeated over several named targets:

```bash
python research/scripts/run_llm_matrix.py \
  --targets opencode-default groq-default openai-default \
  --continue-on-error \
  -- litqa \
  --paper-dir /DATA/paperqa/litqa2/eval \
  --index-name litqa2_eval \
  --split eval \
  --limit 10
```

Each target receives a distinct run ID while all records append to the same baseline
JSONL file.

## Adding another provider

If it is already supported by LiteLLM, normally only add a provider entry with
`adapter: litellm` and its prefix. If it exposes an OpenAI-compatible endpoint, use
`adapter: openai_compatible`, set `api_base`, and name the API-key environment variable.

A new Python adapter is only required when the provider cannot be expressed through
those two transport styles. Register it through `ProviderRegistry`; the LitQA2 and
PaperQA execution code remains unchanged.

## Output contract

Default output:

```text
results/
├── baseline.jsonl
└── traces/<run_id>/
    ├── run_config.json
    ├── runner.log
    ├── summary.json
    └── <question_id>.trace.jsonl
```

`baseline.jsonl` contains one end-to-end record per question. The trace files contain
agent actions, tool arguments/observations, state/evidence snapshots, and provider-
independent LiteLLM call records including token/cost fields when exposed.

## Role overrides are not the homogeneous baseline

The CLI also accepts `--agent-target`, `--summary-target`, and `--answer-target` for
future ablations. If any role resolves to a different model, the output configuration
is recorded as `multi_llm_roles` rather than `homogeneous_llm`. For the reproduction
baseline, do not use these overrides: one target should drive all three LLM roles.

## Jev decision path

After preparing the same LitQA2 questions and article index for both conditions,
add `--decision-backend jev --evidence-relevance-backend jev` to opt into the
heterogeneous path. The default for both flags is `llm`. Set `TYPESAFE_API_KEY`
in the process environment; do not place its value in run metadata. The result
uses `configuration: heterogeneous_static` and the same LitQA2 evaluator and
trace layout as the baseline. Jev operations are separate `jev_call` or
`jev_call_failed` trace events. The per-question record includes `jev_calls`,
`jev_cost_usd`, `jev_input_tokens`, `jev_output_tokens`, and whether the cost
total is complete. If any cost is unknown, `total_cost_usd` is null.

## Groq key pool and adaptive throttling

Groq can use one primary key plus a comma-separated pool:

```dotenv
GROQ_API_KEY=gsk_primary
GROQ_API_KEYS=gsk_secondary,gsk_tertiary,gsk_fourth
```

The runner deduplicates keys and exposes pool secrets to LiteLLM only through
process-local environment-variable references. Raw keys are not written to traces or
run metadata.

### Choose the quota scope explicitly

If all keys belong to one Groq organization, use the conservative shared mode:

```dotenv
GROQ_QUOTA_SCOPE=shared
GROQ_TPM=8000
GROQ_RPM=30
GROQ_RATE_LIMIT_HEADROOM=0.80
```

If the keys are known to have independent organization quotas, use per-key mode:

```dotenv
GROQ_QUOTA_SCOPE=per_key
GROQ_TPM_PER_KEY=8000
GROQ_RPM_PER_KEY=30
GROQ_RATE_LIMIT_HEADROOM=0.80
```

With four independent keys and 80% headroom, every LiteLLM deployment receives a
6400 TPM / 24 RPM local budget, while fhlmi receives one aggregate global limit of
25600 TPM / 96 RPM. The fhlmi limiter is process-global and weighted by estimated
prompt/completion tokens, so the answer, summary, and agent model roles share the same
budget and **wait before dispatch** instead of bursting until Groq rejects the calls.

Do not set `per_key` merely because multiple key strings exist. It is only correct when
their quotas are actually independent. A Groq 429 includes an organization ID, which
is useful for verifying this from smoke-run logs.

### 429 recovery

LiteLLM still handles normal retry/rotation and per-deployment cooldowns. If the entire
pool is exhausted and LiteLLM finally raises a `RateLimitError`, the baseline runner has
a process-local recovery wrapper that:

1. extracts Groq's `Please try again in ...s` / `Retry-After` value;
2. waits for that interval;
3. re-enters the Router so it can select a healthy key again;
4. uses deterministic exponential waiting only when the provider supplies no delay.

Configure it with:

```dotenv
GROQ_RATE_LIMIT_RECOVERY_ATTEMPTS=12
GROQ_RATE_LIMIT_DEFAULT_WAIT_SECONDS=5
GROQ_RATE_LIMIT_MAX_WAIT_SECONDS=60
```

This scheduling layer does not change PaperQA prompts, tools, evidence, or answers. It
only controls when an unchanged LLM request is dispatched. Provider fallback to another
model/service remains disabled because silently switching providers would contaminate a
baseline run.

Inspect the resolved pool without printing secrets:

```bash
UV_PROJECT_ENVIRONMENT=.venv-baseline \
uv run --no-dev --group baseline python - <<'PY'
from dotenv import load_dotenv
from research.baseline.config import ModelCatalog

load_dotenv('.env')
t = ModelCatalog.load('research/configs/model_targets.toml').resolve_inline(
    provider_name='groq',
    model='openai/gpt-oss-120b',
)
cfg = t.router_config(retries=3)
print(t.public_metadata())
print('deployments:', len(cfg['model_list']))
print('lmi rate_limit:', cfg.get('rate_limit'))
print('lmi request_limit:', cfg.get('request_limit'))
PY
```
