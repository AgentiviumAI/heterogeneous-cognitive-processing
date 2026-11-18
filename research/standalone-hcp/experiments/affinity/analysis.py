import argparse
import csv
import json
import math
import statistics
from pathlib import Path
import numpy as np
import yaml
from .common import digest, file_hash, write_new
from benchmarks.scifact.evaluator import ranking_metrics

MODELS = ['qwen3-8b', 'd1-3b']

def bootstrap_pair(a, b, samples, seed):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert len(a) == len(b) and len(a) > 0
    indices = np.random.default_rng(seed).integers(0, len(a), size=(samples, len(a)))
    qa, qb = a[indices].mean(axis=1), b[indices].mean(axis=1)
    interval = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]
    return {'qwen': float(a.mean()), 'd1': float(b.mean()), 'difference_d1_minus_qwen': float((b - a).mean()), 'qwen_ci95': interval(qa), 'd1_ci95': interval(qb), 'difference_ci95': interval(qb - qa), 'paired_units': len(a)}

def latency_stats(rows, dataset, manifests, config):
    subset = [r for r in rows if r['dataset'] == dataset]
    result = {}
    for model in MODELS:
        group = [r for r in subset if r['processor'] == model]
        times = [r['latency_ms'] for r in group]
        inputs = [r['input_tokens'] for r in group if r['input_tokens'] is not None]
        outputs = [r['output_tokens'] for r in group if r['output_tokens'] is not None]
        result[model] = {'median_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95)), 'requests': len(group), 'invalid_requests': sum(not r['valid'] for r in group), 'invalid_rate': float(np.mean([not r['valid'] for r in group])), 'errors': dict((name, sum(r['error'] is not None and r['error']['type'] == name for r in group)) for name in sorted({r['error']['type'] for r in group if r['error']})), 'mean_input_tokens': float(np.mean(inputs)) if inputs else None, 'mean_output_tokens': float(np.mean(outputs)) if outputs else None, 'mean_input_preparation_ms': float(np.mean([r['input_preparation_ms'] for r in group])), 'median_harness_cpu_seconds': float(np.median([r['harness_cpu_seconds'] for r in group])), 'median_container_cpu_usage_usec_delta': float(np.median([r['container_cpu_usage_usec_delta'] for r in group if r['container_cpu_usage_usec_delta'] is not None])) if any(r['container_cpu_usage_usec_delta'] is not None for r in group) else None}
    units = {}
    for row in subset:
        unit = row['query_id'] if dataset == 'scifact' else row['task_id']
        units.setdefault(unit, {m: [] for m in MODELS})[row['processor']].append(row['latency_ms'])
    ordered = sorted(units)
    a = np.asarray([np.median(units[k][MODELS[0]]) for k in ordered])
    b = np.asarray([np.median(units[k][MODELS[1]]) for k in ordered])
    indices = np.random.default_rng(config['seed'] + 2).integers(0, len(a), size=(config['bootstrap_samples'], len(a)))
    ratios = np.median(a[indices], axis=1) / np.median(b[indices], axis=1)
    result['speedup_qwen_over_d1'] = result[MODELS[0]]['median_ms'] / result[MODELS[1]]['median_ms']
    result['cluster_median_speedup'] = float(np.median(a) / np.median(b))
    result['cluster_median_speedup_ci95'] = np.percentile(ratios, [2.5, 97.5]).tolist()
    result['median_paired_latency_difference_ms'] = float(np.median(b - a))
    result['latency_bootstrap_units'] = 'query clusters' if dataset == 'scifact' else 'BFCL example clusters'
    return result

def calculate(rows, manifests, config):
    indexed = {(r['dataset'], r['task_id'], r['processor'], r['pass']): r for r in rows}
    assert len(indexed) == len(rows), 'duplicate trace key'
    expected = sum(len(m['examples']) for m in manifests.values()) * 2 * config['passes']
    assert len(rows) == expected, (len(rows), expected)
    for dataset, manifest in manifests.items():
        for example in manifest['examples']:
            for model in MODELS:
                for measurement_pass in range(config['passes']):
                    row = indexed[(dataset, example['task_id'], model, measurement_pass)]
                    assert row['input_state_hash'] == example['input_state_hash'] == digest(row['input_state'])
    bfcl_units, per_pass = {m: [] for m in MODELS}, {}
    strata = {}
    no_tool = {m: [] for m in MODELS}
    for model in MODELS:
        per_pass[model] = []
        for measurement_pass in range(config['passes']):
            decisions = [indexed[('bfcl', e['task_id'], model, measurement_pass)]['correct'] for e in manifests['bfcl']['examples']]
            per_pass[model].append(float(np.mean(decisions)))
        for example in manifests['bfcl']['examples']:
            values = [indexed[('bfcl', example['task_id'], model, p)]['correct'] for p in range(config['passes'])]
            bfcl_units[model].append(float(np.mean(values)))
            category = example['category']
            strata.setdefault(category, {m: [] for m in MODELS})[model].append(float(np.mean(values)))
            if example['expected_decision'] == 'NO_TOOL':
                no_tool[model].append(float(np.mean(values)))
    bfcl_quality = bootstrap_pair(bfcl_units[MODELS[0]], bfcl_units[MODELS[1]], config['bootstrap_samples'], config['seed'])
    bfcl_quality['per_pass_accuracy'] = per_pass
    bfcl_quality['strata'] = {k: bootstrap_pair(v[MODELS[0]], v[MODELS[1]], config['bootstrap_samples'], config['seed']) for k, v in strata.items()}
    bfcl_quality['no_tool'] = bootstrap_pair(no_tool[MODELS[0]], no_tool[MODELS[1]], config['bootstrap_samples'], config['seed'])
    bfcl_quality['always_first_tool_accuracy'] = float(np.mean([e['expected_decision'] == 'T0' for e in manifests['bfcl']['examples']]))
    bfcl_quality['always_no_tool_accuracy'] = float(np.mean([e['expected_decision'] == 'NO_TOOL' for e in manifests['bfcl']['examples']]))
    query_rows, metrics = [], ['ndcg10', 'mrr10', 'recall10']
    for query in manifests['scifact']['queries']:
        qid = query['query_id']
        documents = [d['document_id'] for d in query['candidates']]
        baseline = ranking_metrics(documents, query['qrels'], pool=documents)
        for model in MODELS:
            for measurement_pass in range(config['passes']):
                scored = [(indexed[('scifact', f'{qid}::{did}', model, measurement_pass)], rank) for rank, did in enumerate(documents)]
                # Invalid decisions go below valid ones, never omitted. Ties retain BM25 ordering.
                scored.sort(key=lambda item: (not item[0]['valid'], -(item[0]['score'] if item[0]['valid'] else -1), item[1]))
                ranking = [row['document_id'] for row, rank in scored]
                outcome = ranking_metrics(ranking, query['qrels'], pool=documents)
                query_rows.append({'query_id': qid, 'processor': model, 'pass': measurement_pass, 'ranking': ranking, 'bm25_baseline': baseline, 'candidate_recall': query['candidate_recall'], 'relevant_in_pool': query['relevant_in_pool'], 'unjudged_in_pool': sum(d not in query['qrels'] for d in documents), **outcome})
    scifact_quality = {}
    for metric in metrics:
        values = {model: [statistics.mean(r[metric] for r in query_rows if r['query_id'] == q['query_id'] and r['processor'] == model) for q in manifests['scifact']['queries']] for model in MODELS}
        scifact_quality[metric] = bootstrap_pair(values[MODELS[0]], values[MODELS[1]], config['bootstrap_samples'], config['seed'] + 1)
    scifact_quality['bm25_baseline'] = {metric: float(np.mean([ranking_metrics([d['document_id'] for d in q['candidates']], q['qrels'])[metric] for q in manifests['scifact']['queries']])) for metric in metrics}
    scifact_quality['candidate_recall_mean'] = float(np.mean([q['candidate_recall'] for q in manifests['scifact']['queries']]))
    scifact_quality['queries_with_no_relevant_in_pool'] = sum(q['relevant_in_pool'] == 0 for q in manifests['scifact']['queries'])
    scifact_quality['queries_with_any_relevant_in_pool'] = sum(q['relevant_in_pool'] > 0 for q in manifests['scifact']['queries'])
    scifact_quality['unjudged_fraction_of_pool'] = sum(r['unjudged_in_pool'] for r in query_rows) / (len(query_rows) * config['scifact_candidates'])
    scifact_quality['conditional_pool_ndcg10'] = {model: float(np.mean([r['pool_conditional_ndcg10'] for r in query_rows if r['processor'] == model and r['pool_conditional_ndcg10'] is not None])) for model in MODELS}
    scifact_quality['per_pass_ndcg10'] = {model: [float(np.mean([r['ndcg10'] for r in query_rows if r['processor'] == model and r['pass'] == p])) for p in range(config['passes'])] for model in MODELS}
    stability = {}
    for dataset, manifest in manifests.items():
        stability[dataset] = {model: sum(len({indexed[(dataset, e['task_id'], model, p)]['decision'] for p in range(config['passes'])}) > 1 for e in manifest['examples']) / len(manifest['examples']) for model in MODELS}
    return {'bfcl': {'quality': bfcl_quality, 'latency': latency_stats(rows, 'bfcl', manifests, config)}, 'scifact': {'quality': scifact_quality, 'latency': latency_stats(rows, 'scifact', manifests, config)}, 'decision_variation_fraction_across_passes': stability, 'total_measured_requests': len(rows), 'confidence_interval_method': f"paired percentile bootstrap, {config['bootstrap_samples']} replicates; average repeated quality per BFCL example or SciFact query before resampling; exploratory, no equivalence test"}, query_rows

def figures(summary, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    destination = root / 'figures'; destination.mkdir(exist_ok=True)
    colors = ['#2864A0', '#C26A20']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for ax, dataset, ylabel in zip(axes, ['bfcl', 'scifact'], ['Tool-selection accuracy', 'Restricted-pool nDCG@10']):
        quality = summary[dataset]['quality'] if dataset == 'bfcl' else summary[dataset]['quality']['ndcg10']
        values = [quality['qwen'], quality['d1']]
        intervals = [quality['qwen_ci95'], quality['d1_ci95']]
        errors = [[max(0, values[i] - intervals[i][0]) for i in range(2)], [max(0, intervals[i][1] - values[i]) for i in range(2)]]
        ax.bar(['Qwen 8B AWQ', 'D1 3B Q8'], values, color=colors, yerr=errors, capsize=4, width=0.65)
        ax.set_ylim(0, 1.12); ax.set_yticks(np.arange(0, 1.01, .2)); ax.set_ylabel(ylabel); ax.set_title('BFCL V3 subset' if dataset == 'bfcl' else 'SciFact, BM25 top-10')
        ax.grid(axis='y', alpha=0.2); ax.set_axisbelow(True)
        for i, value in enumerate(values): ax.text(i, value + 0.035, f'{value:.3f}', ha='center')
    fig.supxlabel('Bars: mean quality; whiskers: 95% cluster bootstrap CI', fontsize=8)
    for ext in ['pdf', 'svg', 'png']: fig.savefig(destination / f'affinity-quality.{ext}', dpi=600)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for ax, dataset in zip(axes, ['bfcl', 'scifact']):
        latency = summary[dataset]['latency']
        values = [latency[m]['median_ms'] for m in MODELS]
        p95 = [latency[m]['p95_ms'] for m in MODELS]
        ax.bar(['Qwen 8B AWQ', 'D1 3B Q8'], values, color=colors, width=0.65)
        ax.errorbar([0, 1], values, yerr=[[0, 0], [max(0, p95[i] - values[i]) for i in range(2)]], fmt='none', color='black', capsize=4)
        ax.set_ylabel('Local HTTP latency (ms)'); ax.set_title('Tool selection' if dataset == 'bfcl' else 'Evidence relevance per pair')
        ax.set_ylim(0, max(p95) * 1.3); ax.grid(axis='y', alpha=0.2); ax.set_axisbelow(True)
        for i, value in enumerate(values): ax.text(i, value + max(p95) * 0.03, f'{value:.1f}', ha='center')
    fig.supxlabel('Bars: median; upper whiskers: P95 (not confidence intervals)', fontsize=8)
    for ext in ['pdf', 'svg', 'png']: fig.savefig(destination / f'affinity-latency.{ext}', dpi=600)
    plt.close(fig)

def report(summary, manifests, root, observations):
    bf, sf = summary['bfcl'], summary['scifact']
    def interval(values): return f'[{values[0]:.4f}, {values[1]:.4f}]'
    def row(name, quality, latency):
        return f"| {name} | {quality['qwen']:.4f} | {quality['d1']:.4f} | {latency[MODELS[0]]['median_ms']:.1f} / {latency[MODELS[0]]['p95_ms']:.1f} | {latency[MODELS[1]]['median_ms']:.1f} / {latency[MODELS[1]]['p95_ms']:.1f} | {quality['difference_d1_minus_qwen']:+.4f}, CI {interval(quality['difference_ci95'])} | {latency['speedup_qwen_over_d1']:.2f}× |"
    matrix = '\n'.join([row('Tool selection: accuracy', bf['quality'], bf['latency']), row('Evidence relevance: nDCG@10', sf['quality']['ndcg10'], sf['latency'])])
    peak = max((r.get('peak_vram_bytes') or 0 for r in observations), default=0)
    lines = [
        '# HCP Cognitive Operation–Processor Affinity Experiments',
        '', '## Scope and operation–processor affinity matrix', '',
        'Same-state micro-characterization on the existing H100 MIG 2g.20gb VM. Qwen3-8B-AWQ/vLLM 0.8.5 and LiquidAI D1-3B Q8_0/native CUDA llama.cpp remain unchanged. No PaperQA, Docker, new GPU model, adaptive routing or agent framework was introduced.', '',
        '| Operation / quality metric | Qwen quality | D1 quality | Qwen median / P95 ms | D1 median / P95 ms | Paired quality difference D1−Qwen | Median latency ratio Qwen/D1 |',
        '|---|---:|---:|---:|---:|---|---:|', matrix, '',
        f"{summary['total_measured_requests']} measured sequential calls, 3 passes. Effective paired sample size: {bf['quality']['paired_units']} BFCL examples and {sf['quality']['ndcg10']['paired_units']} SciFact queries, not the repeated request count. Smoke and warm-up each used 10 examples per dataset per processor and are excluded.", '',
        '## BFCL selection and annotation semantics', '',
        f"BFCL V3 dataset revision `{manifests['bfcl']['revision']}`. Frozen stratified counts: `{manifests['bfcl']['category_counts']}`. AST-reference tool keys and executable reference callable names supply exact selection labels. Official irrelevance-category semantics supply NO_TOOL labels where reference-answer files are absent. Multiple-function categories have one expected call among several candidates; parallel and multi-turn planning are excluded.", '',
        'Model input contains only the original single-turn messages, candidate names, descriptions and parameter schemas, plus the common bounded selection rubric. Gold arguments, labels and references are never sent. Synthetic tool IDs preserve exact original identities. This evaluates tool selection only, not arguments, executable correctness or the full BFCL leaderboard.', '',
        f"Always-first-tool baseline: {bf['quality']['always_first_tool_accuracy']:.3f}; always-NO_TOOL baseline: {bf['quality']['always_no_tool_accuracy']:.3f}. Simple categories have only one candidate and can exhibit a ceiling effect; inspect strata rather than interpreting the pooled result as a universal tool-use score.", '',
        f"NO_TOOL accuracy Qwen/D1: {bf['quality']['no_tool']['qwen']:.4f}/{bf['quality']['no_tool']['d1']:.4f}; paired CI {interval(bf['quality']['no_tool']['difference_ci95'])}.", '',
        '## SciFact candidate coverage and ranking', '',
        f"Official BEIR archive SHA256 `{manifests['scifact']['archive_sha256']}`; {manifests['scifact']['corpus_count']} full-corpus documents. 100 test claims were selected by seeded ID hashing without filtering for retrieval success. BM25Okapi CPU retrieved top 10; title/text prefixes of 240/1600 Unicode characters were frozen identically for both processors. Retrieval used the full untruncated corpus.", '',
        f"Mean candidate recall: {sf['quality']['candidate_recall_mean']:.4f}; {sf['quality']['queries_with_no_relevant_in_pool']}/100 claims have no judged relevant candidate and remain in all primary averages. Unjudged fraction of candidates: {sf['quality']['unjudged_fraction_of_pool']:.3f}.", '',
        'Both processors choose the same integer relevance level 0–3, explicitly counting supporting and refuting evidence as relevant. D1 uses native choice rather than a probability-weighted score to match Qwen output granularity. Ties preserve the frozen BM25 order. Invalid scores are placed below valid scores and are not silently dropped.', '',
        'nDCG uses all official query qrels for IDCG, so missing relevant documents limit attainable quality. Conventional qrel evaluation assigns zero gain to unjudged documents; this does not establish that they are verified negatives. Conditional pool nDCG is diagnostic only and excludes zero-coverage queries explicitly.', '',
        f"BM25 baseline: `{sf['quality']['bm25_baseline']}`. Qwen/D1 MRR@10: {sf['quality']['mrr10']['qwen']:.4f}/{sf['quality']['mrr10']['d1']:.4f}, paired CI {interval(sf['quality']['mrr10']['difference_ci95'])}. Recall@10 is unchanged by reranking because all ten candidates are retained; it is a retrieval coverage measure, not evidence of reranker improvement.", '',
        'These are restricted-pool processor comparisons, not full-corpus BEIR leaderboard scores.', '',
        '## Measurement, failures and reproducibility', '',
        'All operation states are hashed; both processors receive the same semantic state and rubric through their native protocols. No outcome-dependent prompt or sample tuning is performed. Each pass shuffles task order deterministically and counterbalances processor order. Concurrency is one. Token preflight uses both backend tokenizers plus a fixed 256-token reserve; actual token usage is saved, and context-limit responses are rejected.', '',
        f"Invalid counts BFCL Qwen/D1: {bf['latency'][MODELS[0]]['invalid_requests']}/{bf['latency'][MODELS[1]]['invalid_requests']}; SciFact: {sf['latency'][MODELS[0]]['invalid_requests']}/{sf['latency'][MODELS[1]]['invalid_requests']}. Invalid BFCL responses count as incorrect. Error types, raw responses, input overhead, HTTP latency, request order, CPU deltas and usage are retained.", '',
        f"Observed peak resident VRAM during recorded invocations: {peak / 1024**3:.2f} GiB. MIG GPU utilization and GPU execution time are unavailable and remain null. Memory is sampled at 0.5s intervals and can miss brief peaks. Container CPU deltas include service/client/monitoring work and are not isolated GPU or model execution costs.", '',
        'Quality CIs use paired cluster percentile bootstrap with 10,000 replicates, averaging repeated quality before resampling. Latency speedup CIs resample paired per-example medians for BFCL and query-cluster medians for SciFact. They are exploratory CIs, not equivalence tests or multiple-comparison-adjusted confirmatory inference. Temperature is zero; repeated passes primarily characterize timing and response stability.', '',
        f"Across-pass decision variation: `{summary['decision_variation_fraction_across_passes']}`.", '',
        'Quality figure bars use paired-cluster marginal 95% CIs; latency bars show median and upper whiskers show P95, not confidence intervals. PDF/SVG vector figures and 600dpi PNGs are supplied. `summary.json`, `affinity-matrix.csv`, `scifact-query-metrics.json`, raw per-request traces, manifests, source hashes and dataset checksums permit recomputation.', '',
        '## Assessment for HCP arXiv v1', '',
        'Supported for inclusion: measured operation-specific quality, paired uncertainty, local decision latency, error rates and the resource footprint under this exact quantized-model/backend configuration. Present both favorable and unfavorable outcomes; a CI crossing zero is inconclusive, not evidence of established quality equivalence.', '',
        'These experiments primarily characterize RQ2. They may illustrate RQ1 operation heterogeneity but do not independently establish the framework definitions. They do not establish RQ3 end-to-end benefit, adaptive-routing benefit, causal specialization effects independent of size/quantization/backend, or universal processor superiority. Existing PaperQA evidence remains the RQ3 case study and is not changed.', '',
        'Additional limits: public benchmark training exposure is unknown; the stratified BFCL subset is not population-weighted; SciFact qrels are incomplete; prefix truncation and ordinal-score ties constrain ranking resolution; CPU/GPU services share one MIG device. Faster micro-decisions must not be described as automatically accelerating a complete workflow.', '',
    ]
    for dataset, quality in [('Tool selection', bf['quality']), ('Evidence relevance', sf['quality']['ndcg10'])]:
        low, high = quality['difference_ci95']
        conclusion = 'D1 quality is lower under this configuration' if high < 0 else 'D1 quality is higher under this configuration' if low > 0 else 'the paired quality difference is inconclusive'
        lines.append(f"- **{dataset}:** {conclusion}; Δ={quality['difference_d1_minus_qwen']:+.4f}, 95% CI {interval(quality['difference_ci95'])}. See the operation-specific latency ratio above for the trade-off.")
    (root / 'HCP-AFFINITY-REPORT.md').write_text('\n'.join(lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/affinity-v1.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config['results'])
    assert (root / 'full/completed.json').exists(), 'Full run must complete before primary analysis.'
    if (root / 'analysis/complete.json').exists():
        print('ANALYSIS_ALREADY_COMPLETE; no artifacts overwritten', flush=True)
        return
    manifests = {name: json.loads((Path(config['manifest_root']) / f'{name}.json').read_text()) for name in ['bfcl', 'scifact']}
    rows = [json.loads(p.read_text()) for p in root.glob('full/*/pass-*/*.json')]
    summary, query_rows = calculate(rows, manifests, config)
    observations = [json.loads(p.read_text()) for p in root.glob('observations/*.json')]
    write_new(root / 'summary.json', summary)
    write_new(root / 'scifact-query-metrics.json', query_rows)
    with (root / 'affinity-matrix.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['operation', 'quality_metric', 'qwen_quality', 'd1_quality', 'difference_d1_minus_qwen', 'ci_low', 'ci_high', 'qwen_median_ms', 'd1_median_ms', 'qwen_p95_ms', 'd1_p95_ms', 'latency_speedup'])
        writer.writeheader()
        for dataset in ['bfcl', 'scifact']:
            quality = summary[dataset]['quality'] if dataset == 'bfcl' else summary[dataset]['quality']['ndcg10']
            latency = summary[dataset]['latency']
            writer.writerow({'operation': 'tool_selection' if dataset == 'bfcl' else 'evidence_relevance', 'quality_metric': 'accuracy' if dataset == 'bfcl' else 'ndcg10', 'qwen_quality': quality['qwen'], 'd1_quality': quality['d1'], 'difference_d1_minus_qwen': quality['difference_d1_minus_qwen'], 'ci_low': quality['difference_ci95'][0], 'ci_high': quality['difference_ci95'][1], 'qwen_median_ms': latency[MODELS[0]]['median_ms'], 'd1_median_ms': latency[MODELS[1]]['median_ms'], 'qwen_p95_ms': latency[MODELS[0]]['p95_ms'], 'd1_p95_ms': latency[MODELS[1]]['p95_ms'], 'latency_speedup': latency['speedup_qwen_over_d1']})
    figures(summary, root)
    report(summary, manifests, root, observations)
    write_new(root / 'analysis/complete.json', {'trace_count': len(rows), 'summary_sha256': file_hash(root / 'summary.json'), 'config_hash': digest(config)})
    print('ANALYSIS_COMPLETE', json.dumps({'bfcl': summary['bfcl']['quality']['difference_d1_minus_qwen'], 'scifact': summary['scifact']['quality']['ndcg10']['difference_d1_minus_qwen']}), flush=True)

if __name__ == '__main__':
    main()
