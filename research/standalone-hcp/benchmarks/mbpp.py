import hashlib
import json
import urllib.request
from pathlib import Path

def prepare(destination, count, revision=None):
    from datasets import load_dataset
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if revision is None:
        info = json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/google-research-datasets/mbpp', timeout=30))
        revision = info['sha']
    dataset = load_dataset('google-research-datasets/mbpp', 'sanitized', split='test', revision=revision)
    # Strip reference implementation immediately; never store it in experiment tasks.
    tasks = [{'id': int(row['task_id']), 'prompt': row['prompt'], 'tests': row['test_list'], 'setup': row.get('test_imports', [])} for row in dataset]
    tasks = sorted(tasks, key=lambda task: task['id'])
    if count not in (None, 'all'):
        tasks = tasks[:int(count)]
    data = json.dumps(tasks, indent=2)
    (destination / 'tasks.json').write_text(data)
    metadata = {'dataset': 'google-research-datasets/mbpp', 'config': 'sanitized', 'split': 'test', 'revision': revision, 'selection': 'complete official sanitized test split; task IDs sorted ascending; no outcome-based filtering' if count in (None, 'all') else 'first task IDs sorted ascending; no outcome-based filtering', 'count': len(tasks), 'sha256': hashlib.sha256(data.encode()).hexdigest()}
    (destination / 'dataset.json').write_text(json.dumps(metadata, indent=2))
    return tasks
