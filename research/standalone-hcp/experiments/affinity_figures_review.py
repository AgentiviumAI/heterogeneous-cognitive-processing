import json
from pathlib import Path
import numpy as np
MODELS=["qwen3-8b","d1-3b"]
def figures(summary, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    destination = root / 'figures-review'; destination.mkdir(exist_ok=True)
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
        for i, value in enumerate(values): ax.text(i, intervals[i][1] + 0.025, f'{value:.3f}', ha='center')
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
        for i, value in enumerate(values): ax.text(i, p95[i] + max(p95) * 0.035, f'{value:.1f}', ha='center')
    fig.supxlabel('Bars: median; upper whiskers: P95 (not confidence intervals)', fontsize=8)
    for ext in ['pdf', 'svg', 'png']: fig.savefig(destination / f'affinity-latency.{ext}', dpi=600)
    plt.close(fig)

root=Path("results/affinity-v1")
figures(json.loads((root/"summary.json").read_text()),root)
