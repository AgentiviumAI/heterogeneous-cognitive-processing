# Standalone HCP operation-affinity experiments

The paper compares Qwen3-8B-AWQ and LiquidAI D1-3B Q8_0 on 200 BFCL V3 tool-selection states and 100 SciFact claims with ten frozen BM25 candidates each. Each processor receives the same semantic operation state. Three measurement passes are paired by example or query.

## Environment and services

From this directory, create the environment described in the root README using `requirements-affinity.lock.txt`.

Serve Qwen at `http://127.0.0.1:8000` using vLLM 0.8.5 and D1 at `http://127.0.0.1:8081` using the native llama.cpp `/v1/systemone` interface. Model revisions, quantization, context limits and the llama.cpp commit are pinned in [`configs/affinity-v1.yaml`](configs/affinity-v1.yaml). Qwen thinking is disabled by the client. Use one active request at a time. The historical deployment used a GPU/MIG environment; memory/utilization telemetry depends on NVML support.

## Dataset acquisition

The harness expects BFCL JSONL files under `datasets/raw/bfcl/` and BEIR SciFact under `datasets/raw/scifact/`, with the original ZIP retained as `datasets/raw/scifact.zip`.

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
import yaml
from pathlib import Path
config = yaml.safe_load(Path('configs/affinity-v1.yaml').read_text())
snapshot_download(
    repo_id='gorilla-llm/Berkeley-Function-Calling-Leaderboard',
    repo_type='dataset', revision=config['bfcl_revision'],
    allow_patterns=['BFCL_v3_*.json', 'possible_answer/**'],
    local_dir='datasets/raw/bfcl',
)
PY
mkdir -p datasets/raw
curl --fail --location \
  https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip \
  --output datasets/raw/scifact.zip
unzip datasets/raw/scifact.zip -d datasets/raw
```

Dataset terms remain those of the upstream providers. The preparation step records hashes and checks the shared context envelope against both live backends before freezing manifests.

## Execution

```bash
.venv/bin/python -m experiments.affinity.prepare
.venv/bin/python -m experiments.affinity.run --mode smoke
.venv/bin/python -m experiments.affinity.run --mode warmup
.venv/bin/python -m experiments.affinity.run --mode full
.venv/bin/python -m experiments.affinity.analysis
```

The released `results/affinity-v1` directory contains aggregate artifacts, not raw request traces. For a new run, copy the config, choose a fresh `results` path and `manifest_root`, and pass `--config` to every command. Analysis of the historical run requires its original raw traces and frozen manifests, which are not included here.

See the [protocol](docs/affinity-experiment.md) for task selection, bootstrap units, telemetry limits and interpretation. MBPP helpers remain as supporting harness code; they are not evidence for the paper's BFCL/SciFact results. The direct Linux evaluator requires Linux seccomp and privilege configuration; it is not used by the affinity tests.
