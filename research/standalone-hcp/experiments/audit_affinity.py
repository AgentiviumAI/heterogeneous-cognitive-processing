import json
from pathlib import Path
from experiments.affinity.common import digest, file_hash, write_new
root=Path('results/affinity-v1')
meta=json.loads((root/'metadata.json').read_text())
assert all(file_hash(p)==h for p,h in meta['source_sha256'].items())
freeze=json.loads(Path('datasets/manifests/affinity-v1/freeze.json').read_text())
assert all(file_hash(p)==h for p,h in freeze['manifest_sha256'].items())
rows=[json.loads(p.read_text()) for p in root.glob('full/*/pass-*/*.json')]
assert len(rows)==7200
ordered=sorted(rows,key=lambda r:r['request_start_utc'])
assert all(a['request_end_utc']<=b['request_start_utc'] for a,b in zip(ordered,ordered[1:])), 'overlapping requests'
assert all(digest(r['input_state'])==r['input_state_hash'] for r in rows)
assert all(not ({'qrels','ground_truth','expected_decision','reference_origin'} & set(r['input_state']['state'])) for r in rows)
summary=json.loads((root/'summary.json').read_text())
maxinputs={m:max((r['input_tokens'] for r in rows if r['processor']==m and r['input_tokens'] is not None),default=None) for m in ['qwen3-8b','d1-3b']}
old=Path('logs/mbpp-before-affinity.sha256').read_text().split()
assert file_hash(old[1])==old[0]
audit={'full_requests':len(rows),'source_hashes_unchanged':True,'manifest_hashes_unchanged':True,'sequential_timestamps_verified':True,'state_hashes_verified':True,'mbpp_summary_unchanged':True,'maximum_reported_input_tokens':maxinputs,'actual_generation_control':{'qwen':'temperature=0, seed=20261014, non-thinking, constrained JSON','d1':'native bounded-choice endpoint; no seed or temperature field sent; experiment seed in traces denotes sampling/order, not a D1 generation parameter'},'official_evaluator_commit':'6ea57973c7a6097fd7c5915698c54c17c5b1b6c8','official_evaluator_sha256':file_hash('datasets/raw/bfcl/official-ast-checker-pinned.py')}
write_new(root/'verification-audit.json',audit)
lines=['','## Verified supplementary detail','','All 7,200 measured records passed source/manifest checksum, state-hash and non-overlapping request timestamp checks. The old MBPP summary checksum is unchanged.','',f'Maximum reported input tokens: `{maxinputs}`. Preflight token counts are surrogate counts plus a fixed reserve, not exact native-template token counts. Backend usage is preserved to check the observed envelope.','',audit['actual_generation_control']['qwen']+'. '+audit['actual_generation_control']['d1']+'. The seed field must not be read as proof that both servers received an identical generation seed. Native protocol, size and quantization differ, so this characterizes complete deployed processor configurations.','',f"Official BFCL evaluator source inspected at commit `{audit['official_evaluator_commit']}`, SHA256 `{audit['official_evaluator_sha256']}`. Exact function identity is evaluated; the full argument checker is not invoked for this bounded operation.",'','| BFCL stratum | n | Qwen accuracy | D1 accuracy | D1−Qwen 95% CI |','|---|---:|---:|---:|---|']
for category,q in summary['bfcl']['quality']['strata'].items():
    lines.append(f"| {category} | {q['paired_units']} | {q['qwen']:.4f} | {q['d1']:.4f} | {q['difference_ci95']} |")
lines+=['','| Operation | Median ratio Qwen/D1 | Cluster-median ratio (95% CI) | Median paired D1−Qwen ms |','|---|---:|---|---:|']
for dataset in ['bfcl','scifact']:
    s=summary[dataset]['latency']
    lines.append(f"| {dataset} | {s['speedup_qwen_over_d1']:.3f} | {s['cluster_median_speedup']:.3f} ({s['cluster_median_speedup_ci95']}) | {s['median_paired_latency_difference_ms']:.3f} |")
lines+=['','The aggregate request-median ratio and the ratio of cluster medians are different estimands. The confidence interval belongs to the latter. BFCL strata are exploratory and unadjusted; related dataset templates may create correlations beyond the example-level bootstrap.','']
with (root/'verification-supplement.md').open('x') as f:f.write('\n'.join(lines))
print(json.dumps(audit))
