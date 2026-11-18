# HCP affinity-v1 protocol

This addition preserves the existing framework and MBPP artifacts. It characterizes RQ2; it does not introduce adaptive routing, redesign HCP, or alter the PaperQA RQ3 case study.

## Frozen scope

E1: 200 BFCL V3 single-turn, single-call decisions. Quotas are 30 simple, 30 live_simple, 20 exec_simple, 40 multiple, 40 live_multiple, 20 irrelevance and 20 live_irrelevance. Multiple-function categories contain several candidates and one expected call, not multi-call planning. Simple examples are retained as requested but have one candidate and a ceiling effect. Category-stratified results and first-tool/NO_TOOL baselines are reported. Ground truth is read only by preparation/evaluation; it is not copied into operation states. Exact duplicate semantic states and ambiguous/multiple references are excluded before inference, with logged reasons.

E2: 100 official SciFact test queries selected using SHA256(seed,original query ID). Full-corpus BM25Okapi CPU retrieves a frozen top-10. Unicode word/casefold tokenization has no stemming or stopword removal; k1=1.5, b=.75, epsilon=.25. Retrieval tie-breaking uses original string document IDs. Neither qrels nor processor scores affect retrieval or query selection. Zero-coverage queries are retained.

Both models use the same discrete ordinal relevance rubric 0–3. D1 uses native choice rather than expected-value score, avoiding a granularity difference that could itself change rankings. Supporting and refuting scientific evidence both count as relevant. Ranking ties retain BM25 order; invalid scores rank after valid ones. Unjudged documents have conventional zero gain for qrel metrics, but are not described as verified negatives. Since all 10 candidates remain, Recall@10 is unchanged by reranking.

## Context and execution

Titles/text use fixed prefix limits of 240/1600 Unicode characters. BFCL schemas are preserved intact. A predeclared operation envelope is checked before freezing: ≤4500 Unicode characters, both backend tokenizer counts plus 256-token reserve below the smaller applicable context. Oversize BFCL rows are excluded before sampling completion; SciFact preparation fails rather than dropping a selected query/pair. Native protocol templates differ, so surrogate counts are preflight bounds; reported actual usage is recorded and context-limit responses rejected. No hidden backend truncation is intended; template effects cannot be eliminated merely by equal semantic state hashes.

Smoke: 10 BFCL examples and top-ranked candidate for 10 distinct SciFact claims, through both processors. Validate shared hashes, exact allowed output extraction and evaluator behavior. This is not prompt optimization. Warmup repeats 10 examples per dataset/model and is excluded. Full run: 3 measurement passes; shuffled task order and counterbalanced model order, one active request at a time. No automatic retries conceal timeout/error rates.

Quality is averaged over repeated passes within BFCL examples or SciFact queries. Paired 95% percentile bootstrap CIs use 10,000 replicates with fixed seed; repeated requests are not counted as independent quality samples. SciFact latency CIs cluster all document pairs by query. BFCL source/template dependencies may remain beyond per-example clustering. CIs are exploratory, not proof of equivalence or confirmatory multiple-comparison-adjusted tests.

## Resources and reproducibility

Reuse existing Qwen/D1 HTTP clients and MIG NVML monitor. Each immutable request trace records full semantic state, hash, model/revision/quantization, order, pass, raw response, validity, error type, usage, latency, input overhead, client CPU and container CPU/memory observations. MIG utilization/GPU execution time remain null when unavailable. Container deltas include client/monitoring work; they are not isolated model CPU costs. Memory sampling can miss brief peaks.

Results are under a new `results/affinity-v1` directory. Existing MBPP summaries are checksummed before/after. Model weights/binaries/services are not changed. Source/config/manifests are frozen before comparative inference; changing them requires a new experiment directory. Partial runs can resume without overwriting completed traces.

```bash
cd /network-volume/hcp-pilot/hcp-experiment
export HF_HOME=/network-volume/hcp-pilot/cache/huggingface
export XDG_CACHE_HOME=/network-volume/hcp-pilot/cache
export MPLCONFIGDIR=/network-volume/hcp-pilot/cache/matplotlib
export TMPDIR=/network-volume/hcp-pilot/tmp
PY=/network-volume/hcp-pilot/venv/bin/python
$PY -m experiments.affinity.prepare
$PY -m pytest -q tests/test_affinity.py
$PY -m experiments.affinity.run --mode smoke
$PY -m experiments.affinity.run --mode warmup
$PY -m experiments.affinity.run --mode full
$PY -m experiments.affinity.analysis
```

Raw BFCL is downloaded with `huggingface_hub.snapshot_download` at the configuration's pinned SHA, using `BFCL_v3_*.json` and `possible_answer/**` patterns. SciFact uses the official BEIR ZIP; its archive and extracted-file checksums are stored in the manifest. The authoritative annotation reference is the pinned BFCL official AST checker. This operation-only evaluation does not use its argument checker as a substitute for tool-selection accuracy.
