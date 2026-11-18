# HCP Cognitive Operation–Processor Affinity Experiments

## Scope and operation–processor affinity matrix

Same-state micro-characterization on the existing H100 MIG 2g.20gb VM. Qwen3-8B-AWQ/vLLM 0.8.5 and LiquidAI D1-3B Q8_0/native CUDA llama.cpp remain unchanged. No PaperQA, Docker, new GPU model, adaptive routing or agent framework was introduced.

| Operation / quality metric | Qwen quality | D1 quality | Qwen median / P95 ms | D1 median / P95 ms | Paired quality difference D1−Qwen | Median latency ratio Qwen/D1 |
|---|---:|---:|---:|---:|---|---:|
| Tool selection: accuracy | 0.9050 | 0.9200 | 175.8 / 240.9 | 51.7 / 113.1 | +0.0150, CI [-0.0250, 0.0600] | 3.40× |
| Evidence relevance: nDCG@10 | 0.6591 | 0.6654 | 198.9 / 222.5 | 68.9 / 87.7 | +0.0063, CI [-0.0262, 0.0395] | 2.89× |

7200 measured sequential calls, 3 passes. Effective paired sample size: 200 BFCL examples and 100 SciFact queries, not the repeated request count. Smoke and warm-up each used 10 examples per dataset per processor and are excluded.

## BFCL selection and annotation semantics

BFCL V3 dataset revision `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`. Frozen stratified counts: `{'simple': 30, 'live_simple': 30, 'exec_simple': 20, 'multiple': 40, 'live_multiple': 40, 'irrelevance': 20, 'live_irrelevance': 20}`. AST-reference tool keys and executable reference callable names supply exact selection labels. Official irrelevance-category semantics supply NO_TOOL labels where reference-answer files are absent. Multiple-function categories have one expected call among several candidates; parallel and multi-turn planning are excluded.

Model input contains only the original single-turn messages, candidate names, descriptions and parameter schemas, plus the common bounded selection rubric. Gold arguments, labels and references are never sent. Synthetic tool IDs preserve exact original identities. This evaluates tool selection only, not arguments, executable correctness or the full BFCL leaderboard.

Always-first-tool baseline: 0.580; always-NO_TOOL baseline: 0.200. Simple categories have only one candidate and can exhibit a ceiling effect; inspect strata rather than interpreting the pooled result as a universal tool-use score.

NO_TOOL accuracy Qwen/D1: 0.5250/0.7500; paired CI [0.0750, 0.3750].

## SciFact candidate coverage and ranking

Official BEIR archive SHA256 `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`; 5183 full-corpus documents. 100 test claims were selected by seeded ID hashing without filtering for retrieval success. BM25Okapi CPU retrieved top 10; title/text prefixes of 240/1600 Unicode characters were frozen identically for both processors. Retrieval used the full untruncated corpus.

Mean candidate recall: 0.7570; 21/100 claims have no judged relevant candidate and remain in all primary averages. Unjudged fraction of candidates: 0.911.

Both processors choose the same integer relevance level 0–3, explicitly counting supporting and refuting evidence as relevant. D1 uses native choice rather than a probability-weighted score to match Qwen output granularity. Ties preserve the frozen BM25 order. Invalid scores are placed below valid scores and are not silently dropped.

nDCG uses all official query qrels for IDCG, so missing relevant documents limit attainable quality. Conventional qrel evaluation assigns zero gain to unjudged documents; this does not establish that they are verified negatives. Conditional pool nDCG is diagnostic only and excludes zero-coverage queries explicitly.

BM25 baseline: `{'ndcg10': 0.6116031547449603, 'mrr10': 0.573734126984127, 'recall10': 0.757}`. Qwen/D1 MRR@10: 0.6425/0.6488, paired CI [-0.0393, 0.0523]. Recall@10 is unchanged by reranking because all ten candidates are retained; it is a retrieval coverage measure, not evidence of reranker improvement.

These are restricted-pool processor comparisons, not full-corpus BEIR leaderboard scores.

## Measurement, failures and reproducibility

All operation states are hashed; both processors receive the same semantic state and rubric through their native protocols. No outcome-dependent prompt or sample tuning is performed. Each pass shuffles task order deterministically and counterbalances processor order. Concurrency is one. Token preflight uses both backend tokenizers plus a fixed 256-token reserve; actual token usage is saved, and context-limit responses are rejected.

Invalid counts BFCL Qwen/D1: 0/0; SciFact: 3/0. Invalid BFCL responses count as incorrect. Error types, raw responses, input overhead, HTTP latency, request order, CPU deltas and usage are retained.

Observed peak resident VRAM during recorded invocations: 14.15 GiB. MIG GPU utilization and GPU execution time are unavailable and remain null. Memory is sampled at 0.5s intervals and can miss brief peaks. Container CPU deltas include service/client/monitoring work and are not isolated GPU or model execution costs.

Quality CIs use paired cluster percentile bootstrap with 10,000 replicates, averaging repeated quality before resampling. Latency speedup CIs resample paired per-example medians for BFCL and query-cluster medians for SciFact. They are exploratory CIs, not equivalence tests or multiple-comparison-adjusted confirmatory inference. Temperature is zero; repeated passes primarily characterize timing and response stability.

Across-pass decision variation: `{'bfcl': {'qwen3-8b': 0.0, 'd1-3b': 0.0}, 'scifact': {'qwen3-8b': 0.0, 'd1-3b': 0.0}}`.

Quality figure bars use paired-cluster marginal 95% CIs; latency bars show median and upper whiskers show P95, not confidence intervals. PDF/SVG vector figures and 600dpi PNGs are supplied. `summary.json`, `affinity-matrix.csv`, `scifact-query-metrics.json`, raw per-request traces, manifests, source hashes and dataset checksums permit recomputation.

## Assessment for HCP arXiv v1

Supported for inclusion: measured operation-specific quality, paired uncertainty, local decision latency, error rates and the resource footprint under this exact quantized-model/backend configuration. Present both favorable and unfavorable outcomes; a CI crossing zero is inconclusive, not evidence of established quality equivalence.

These experiments primarily characterize RQ2. They may illustrate RQ1 operation heterogeneity but do not independently establish the framework definitions. They do not establish RQ3 end-to-end benefit, adaptive-routing benefit, causal specialization effects independent of size/quantization/backend, or universal processor superiority. Existing PaperQA evidence remains the RQ3 case study and is not changed.

Additional limits: public benchmark training exposure is unknown; the stratified BFCL subset is not population-weighted; SciFact qrels are incomplete; prefix truncation and ordinal-score ties constrain ranking resolution; CPU/GPU services share one MIG device. Faster micro-decisions must not be described as automatically accelerating a complete workflow.

- **Tool selection:** the paired quality difference is inconclusive; Δ=+0.0150, 95% CI [-0.0250, 0.0600]. See the operation-specific latency ratio above for the trade-off.
- **Evidence relevance:** the paired quality difference is inconclusive; Δ=+0.0063, 95% CI [-0.0262, 0.0395]. See the operation-specific latency ratio above for the trade-off.