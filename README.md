# Heterogeneous Cognitive Processing

Source code accompanying **Heterogeneous Cognitive Processing: A Systems Perspective on Agentic Workloads** by Thanh Hoang Le Hai, Huy Nguyen Tuan, Nhan Nguyen Phuc, Duong Nguyen Anh, Thu Ho Thi Minh, and Nam Thoai.

This release contains the PaperQA2/Jev integration and the standalone operation-affinity experiments described in the preprint. An arXiv identifier is not yet available.

## What is included

| Study | Implementation | Recorded artifacts |
| --- | --- | --- |
| BFCL tool selection and SciFact evidence relevance | [`research/standalone-hcp`](research/standalone-hcp/) | Pinned configuration, aggregate quality/latency results and verification notes |
| Homogeneous PaperQA2 versus static PaperQA2 + Jev | [`src/paperqa`](src/paperqa/), [`research/baseline`](research/baseline/), [`research/decision_model`](research/decision_model/) | Episode metrics for the 96 paired LitQA2 IDs under each of three base models |

The workflow configurations change decision policy and action interfaces alongside processor assignment. The results characterize complete configurations; they do not isolate a causal processor speedup. Letter matching is a pilot proxy, not official LitQA2 grading.

## PaperQA2/Jev setup

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/AgentiviumAI/heterogeneous-cognitive-processing.git
cd heterogeneous-cognitive-processing
uv sync --no-default-groups --group baseline
uv run --no-sync python research/scripts/run_llm_baseline.py --help
uv run --no-sync python -m research.decision_model.run_jev_pilot --help
```

Provide provider credentials through environment variables, for example `GROQ_API_KEY` and `TYPESAFE_API_KEY`. Model endpoints and environment-variable names are configured in [`research/configs/model_targets.toml`](research/configs/model_targets.toml). The historical `pilot.yaml` describes the earlier five-question preparation gate, not the paper's 96-question run.

For a small open-full-text integration pilot:

```bash
uv run --no-sync python research/datasets/litqa2/prepare_pilot.py --output-dir /tmp/litqa2-pilot
uv run --no-sync python research/datasets/litqa2/prepare_corpus.py \
  --questions /tmp/litqa2-pilot/questions.jsonl --output-dir /tmp/hcp-litqa2-corpus
uv run --no-sync python -m research.decision_model.run_jev_pilot \
  --questions /tmp/litqa2-pilot/questions.jsonl \
  --answer-key /tmp/litqa2-pilot/answer_key.jsonl \
  --corpus /tmp/hcp-litqa2-corpus --timeout 600 --max-timesteps 12
```

See the [baseline runner guide](research/docs/baseline-runner.md) for homogeneous execution and configurable paired runs. Article access and provider availability may differ from the original run. The original 96-question corpus, raw answer text and evidence traces are not redistributed, so the small public pilot is not an exact replay of those workflow experiments.

## Operation-affinity setup

Use a separate environment and run commands from `research/standalone-hcp`:

```bash
cd research/standalone-hcp
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-affinity.lock.txt
.venv/bin/python -m pytest -q tests/test_affinity.py
```

The [affinity guide](research/standalone-hcp/README.md) explains dataset acquisition, model services and execution. The [protocol](research/standalone-hcp/docs/affinity-experiment.md) specifies contracts, sampling and analysis. Rerunning inference needs the pinned Qwen and D1 model services; tests do not call them.

## Recorded workflow metrics

```bash
python3 scripts/summarize_workflows.py
```

This offline command reproduces the paper's completion counts, completed letter-match counts and all-attempt latency medians from the released metrics. It does not call model APIs. See [`results/README.md`](results/README.md) for coverage and exclusions.

## Provenance and licenses

This is a curated source snapshot, without the development repository's Git history. [`SOURCE_PROVENANCE.json`](SOURCE_PROVENANCE.json) records the source commit, source-file checksums and the supplied preprint's checksum. Later scientific-workflow experiments are outside this release.

PaperQA2 originated at [Future House](https://github.com/Future-House/paper-qa). Its source and the Agentivium research additions use [Apache-2.0](LICENSE), except where a file or package states otherwise. The original PaperQA citation is retained in [`CITATION-PaperQA.cff`](CITATION-PaperQA.cff). LitQA2 data is distributed under its separate [CC BY-SA 4.0 license](research/datasets/litqa2/LICENSE). BFCL, SciFact and model weights are obtained separately and retain their upstream terms. This source release does not distribute model weights or hosted decision-service implementations.
