# Recorded results

The five `metrics.jsonl` files contain 576 episodes: 96 paired LitQA2 question IDs for each of GPT-OSS 120B, Qwen3-235B and Qwen3.5-122B. GPT-OSS stores both modes in `viettel-all`; Qwen stores the modes separately. Original directory names are retained as experiment identifiers.

The records preserve question IDs, terminal status, pilot grading flags, latency, call counts, token counts, available costs and model/source identifiers. Question text, reference letters, generated answers, article excerpts, tool histories, raw traces, error strings and local paths are omitted. Missing costs remain null.

Run `python3 scripts/summarize_workflows.py` from the repository root to reconstruct the workflow table. `correct` and letter matching are pilot measures, not official LitQA2 grading. These metrics support aggregate checks, not a full trace audit or reconstruction of the original article corpus.

Operation-affinity aggregates and verification notes are under `research/standalone-hcp/results/affinity-v1/`. Raw datasets, model weights and inference traces are not included.
