
## Verified supplementary detail

All 7,200 measured records passed source/manifest checksum, state-hash and non-overlapping request timestamp checks. The old MBPP summary checksum is unchanged.

Maximum reported input tokens: `{'qwen3-8b': 998, 'd1-3b': 923}`. Preflight token counts are surrogate counts plus a fixed reserve, not exact native-template token counts. Backend usage is preserved to check the observed envelope.

temperature=0, seed=20261014, non-thinking, constrained JSON. native bounded-choice endpoint; no seed or temperature field sent; experiment seed in traces denotes sampling/order, not a D1 generation parameter. The seed field must not be read as proof that both servers received an identical generation seed. Native protocol, size and quantization differ, so this characterizes complete deployed processor configurations.

Official BFCL evaluator source inspected at commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`, SHA256 `2aae7a68461a8f76c0be3894c8901b66b56967a1989d3ab066051e3fb97f1538`. Exact function identity is evaluated; the full argument checker is not invoked for this bounded operation.

| BFCL stratum | n | Qwen accuracy | D1 accuracy | D1−Qwen 95% CI |
|---|---:|---:|---:|---|
| simple | 30 | 1.0000 | 1.0000 | [0.0, 0.0] |
| live_simple | 30 | 1.0000 | 0.8667 | [-0.2666666666666667, -0.033333333333333326] |
| exec_simple | 20 | 1.0000 | 1.0000 | [0.0, 0.0] |
| multiple | 40 | 1.0000 | 1.0000 | [0.0, 0.0] |
| live_multiple | 40 | 1.0000 | 0.9500 | [-0.125, 0.0] |
| irrelevance | 20 | 0.6000 | 0.7500 | [-0.050000000000000044, 0.35000000000000003] |
| live_irrelevance | 20 | 0.4500 | 0.7500 | [0.050000000000000044, 0.55] |

| Operation | Median ratio Qwen/D1 | Cluster-median ratio (95% CI) | Median paired D1−Qwen ms |
|---|---:|---|---:|
| bfcl | 3.398 | 3.425 ([3.248269610805196, 3.725253767830462]) | -133.326 |
| scifact | 2.886 | 2.894 ([2.8646118032247205, 2.936495792570321]) | -129.357 |

The aggregate request-median ratio and the ratio of cluster medians are different estimands. The confidence interval belongs to the latter. BFCL strata are exploratory and unadjusted; related dataset templates may create correlations beyond the example-level bootstrap.
