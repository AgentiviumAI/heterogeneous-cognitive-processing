"""Resumable paired MBPP pilot. Run from project root: python -m experiments.compare."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import uuid
import time
import yaml
from benchmarks.mbpp import prepare
from evaluators.process_runner import evaluate
from processors.qwen import Qwen
from processors.d1 import D1
from processors.schema import Decision
from workflows.code_repair import state_for, decision
from .monitor import Monitor

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)

def cached(path, operation):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    value = operation()
    save(path, value)
    return value

def summarize(rows, initial, replays, config):
    variants = {}
    for variant in config['variants']:
        arm = [r for r in rows if r['variant'] == variant]
        variants[variant] = {'tasks': len(arm), 'initial_passes': sum(r['initial_passed'] for r in arm), 'success_within_3_attempts': sum(r['passed'] for r in arm), 'success_rate': sum(r['passed'] for r in arm) / len(arm), 'model_calls': sum(r['model_calls'] for r in arm), 'mean_logical_workflow_seconds': statistics.mean(r['logical_workflow_seconds'] for r in arm), 'decision_latency_seconds': [d['latency_seconds'] for r in arm for d in r['decisions']], 'extra_successful_repairs_over_initial': sum(r['passed'] and not r['initial_passed'] for r in arm)}
    paired = {}
    baseline = {r['id']: r['passed'] for r in rows if r['variant'] == 'A'}
    for variant in config['variants'][1:]:
        arm = {r['id']: r['passed'] for r in rows if r['variant'] == variant}
        paired[variant + '_vs_A'] = {'wins': sum(arm[k] and not baseline[k] for k in arm), 'losses': sum(baseline[k] and not arm[k] for k in arm), 'ties': sum(arm[k] == baseline[k] for k in arm)}
    return {'config': config, 'variants': variants, 'paired': paired, 'initial_failure_count': sum(not row['evaluation']['passed'] for row in initial), 'replay_count': len(replays), 'limitations': ['Success within three attempts is not statistical pass@3.', 'Logical latency for policies sharing counterfactual generations is reconstructed from shared artifacts; it is distinct from independently executed wall time.', 'Rule policy is a fixed comparator, not a gold label.', 'Model size, quantization, prompt/protocol and serving backend differ.', 'MIG GPU utilization and GPU execution time are not inferred from wall time.']}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/local-gpu.yaml')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config['results'])
    root.mkdir(parents=True, exist_ok=True)
    tasks_path = root / 'dataset/tasks.json'
    if not tasks_path.exists():
        prepare(root / 'dataset', config['tasks'], config.get('dataset_revision'))
    tasks = json.loads(tasks_path.read_text())
    dataset_metadata = json.loads((root / 'dataset/dataset.json').read_text())
    dataset_sha = hashlib.sha256(tasks_path.read_bytes()).hexdigest()
    assert dataset_metadata['sha256'] == dataset_sha, 'Dataset task file hash mismatch.'
    assert dataset_metadata['revision'] == config.get('dataset_revision'), 'Dataset revision differs from immutable experiment config.'
    if args.prepare_only:
        print('PREPARED', len(tasks), flush=True)
        return
    initial_source_root = Path(config.get('initial_generation_source', ''))
    source_tasks, reusable_initial_source = {}, False
    source_dataset_path = initial_source_root / 'dataset/tasks.json'
    source_metadata_path = initial_source_root / 'metadata.json'
    source_dataset_metadata_path = initial_source_root / 'dataset/dataset.json'
    if source_dataset_path.exists() and source_metadata_path.exists() and source_dataset_metadata_path.exists():
        source_config = json.loads(source_metadata_path.read_text()).get('config', {})
        source_dataset_metadata = json.loads(source_dataset_metadata_path.read_text())
        reusable_initial_source = source_dataset_metadata.get('revision') == config.get('dataset_revision') and source_config.get('qwen_revision') == config.get('qwen_revision') and source_config.get('qwen_model') == config.get('qwen_model') and source_config.get('max_output_tokens') == config.get('max_output_tokens')
        if reusable_initial_source:
            source_tasks = {int(item['id']): item for item in json.loads(source_dataset_path.read_text())}
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (root / 'metadata.json').exists():
        assert json.loads((root / 'metadata.json').read_text())['config_sha256'] == config_sha, 'Config changed; use a new result directory.'
    else:
        source_paths = [p for directory in ['benchmarks', 'evaluators', 'experiments', 'processors', 'workflows'] for p in Path(directory).glob('**/*.py')]
        source_paths += [Path('pyproject.toml'), Path('configs/mbpp-extended-v1.yaml')]
        source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths) if p.is_file() and not p.name.startswith('._')}
        packages = {}
        for package in ['httpx', 'pydantic', 'datasets', 'nvidia-ml-py', 'PyYAML', 'numpy', 'matplotlib']:
            try: packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: packages[package] = None
        try: commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        except Exception: commit = None
        config_file = Path(args.config)
        config_copy = root / 'experiment-config.yaml'
        if config_copy.exists():
            assert config_copy.read_bytes() == config_file.read_bytes(), 'Persisted config differs; use a new result directory.'
        else:
            config_copy.write_text(config_file.read_text())
        evaluator = {'backend': config.get('evaluator_backend', os.environ.get('HCP_EVALUATOR_BACKEND', 'native')), 'image': config.get('evaluator_image')}
        if evaluator['image']:
            try: evaluator['image_id'] = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', evaluator['image']], text=True).strip()
            except Exception: evaluator['image_id'] = None
        save(root / 'metadata.json', {'config_sha256': config_sha, 'config_file_sha256': hashlib.sha256(config_file.read_bytes()).hexdigest(), 'dataset': dataset_metadata, 'source_sha256': source_hashes, 'source_commit_sha': commit, 'started': time.time(), 'platform': platform.platform(), 'python': platform.python_version(), 'packages': packages, 'evaluator': evaluator, 'model_backend_identifiers': {k: config.get(k) for k in ['qwen_model','qwen_revision','qwen_backend','d1_revision','d1_backend','llama_commit','d1_quantization']}, 'config': config})
    qwen, d1 = Qwen(config), D1(config)
    qwen.client.get('/health').raise_for_status()
    d1.client.get('/health').raise_for_status()
    monitor = Monitor()
    monitor.start()
    started = time.monotonic()
    initial, replays, rows = [], [], []
    try:
        for task in tasks:
            task_root = root / str(task['id'])
            def make_initial():
                source = initial_source_root / str(task['id']) / 'initial.json'
                reused = reusable_initial_source and source.exists() and source_tasks.get(int(task['id'])) == task
                if reused:
                    generation = json.loads(source.read_text())['generation']
                    generation['request'] = qwen.generation_request(task, task['id'])
                    generation['request_trace_reconstructed'] = True
                else:
                    generation = qwen.generate(task, task['id'])
                return {'generation': generation, 'evaluation': evaluate(generation['code'], task, config['timeout_seconds'], config.get('evaluator_backend'), config.get('memory_mb', 512), config.get('evaluator_image', 'hcp-mbpp-evaluator:py313-numpy226')), 'source_initial_artifact': str(source) if reused else None, 'source_initial_sha256': hashlib.sha256(source.read_bytes()).hexdigest() if reused else None}
            first = cached(task_root / 'initial.json', make_initial)
            initial.append(first)
            seed = task['id'] * 10 + 1
            initial_state = state_for(task, first['generation']['code'], first['evaluation'])
            initial_decisions = {}
            counterfactuals = {}
            if not first['evaluation']['passed']:
                # All four policies see the same saved failure state before any repairs.
                for variant in config['variants']:
                    raw = cached(task_root / f'replay-{variant}.json', lambda v=variant: decision(v, qwen, d1, initial_state, seed).model_dump())
                    initial_decisions[variant] = Decision.model_validate(raw)
                # Outcome-based counterfactuals, not hand-assigned strategy gold labels.
                for action in ['REPAIR', 'REGENERATE']:
                    def generate_branch(action=action):
                        generation = qwen.generate(task, seed, first['generation']['code'], first['evaluation']['feedback'], Decision(action=action))
                        return {'generation': generation, 'evaluation': evaluate(generation['code'], task, config['timeout_seconds'], config.get('evaluator_backend'), config.get('memory_mb', 512), config.get('evaluator_image', 'hcp-mbpp-evaluator:py313-numpy226'))}
                    counterfactuals[action] = cached(task_root / f'counterfactual-{action}.json', generate_branch)
                replays.append({'id': task['id'], 'decisions': {k: v.model_dump() for k, v in initial_decisions.items()}, 'outcomes': {k: v['evaluation']['passed'] for k, v in counterfactuals.items()}})
            for variant in config['variants']:
                arm_path = task_root / f'arm-{variant}.json'
                if arm_path.exists():
                    row = json.loads(arm_path.read_text())
                else:
                    current = first
                    attempts = [first]
                    decisions = []
                    while not current['evaluation']['passed'] and len(attempts) < config['max_attempts']:
                        step = len(attempts)
                        choice = initial_decisions[variant] if step == 1 else decision(variant, qwen, d1, state_for(task, current['generation']['code'], current['evaluation']), task['id'] * 10 + step)
                        decisions.append(choice.model_dump())
                        if step == 1 and choice.action in counterfactuals:
                            current = counterfactuals[choice.action]
                        else:
                            generation = qwen.generate(task, task['id'] * 10 + step, current['generation']['code'], current['evaluation']['feedback'], choice)
                            current = {'generation': generation, 'evaluation': evaluate(generation['code'], task, config['timeout_seconds'], config.get('evaluator_backend'), config.get('memory_mb', 512), config.get('evaluator_image', 'hcp-mbpp-evaluator:py313-numpy226'))}
                        attempts.append(current)
                    row = {'id': task['id'], 'variant': variant, 'initial_passed': first['evaluation']['passed'], 'passed': current['evaluation']['passed'], 'attempt_count': len(attempts), 'attempts': attempts, 'decisions': decisions, 'model_calls': len(attempts) + sum(d['model_calls'] for d in decisions), 'logical_workflow_seconds': sum(a['generation']['latency_seconds'] + a['evaluation']['latency_seconds'] for a in attempts) + sum(d['latency_seconds'] for d in decisions)}
                    save(arm_path, row)
                rows.append(row)
            save(root / 'summary.json', summarize(rows, initial, replays, config))
            save(root / 'replays.json', replays)
            print('TASK_DONE', task['id'], 'initial_pass', first['evaluation']['passed'], flush=True)
    finally:
        samples = monitor.finish()
        save(root / 'gpu-samples.json', samples)
        invocation = {'invocation_id': str(uuid.uuid4()), 'started_unix_seconds': time.time() - (time.monotonic() - started), 'suite_wall_seconds_this_invocation': time.monotonic() - started, 'peak_vram_bytes': max((s['used_bytes'] for s in samples), default=None), 'gpu_time_seconds': None, 'gpu_monitor_unavailable_reason': getattr(monitor, 'unavailable_reason', None)}
        invocation_path = root / 'run-invocations' / f"{invocation['invocation_id']}.json"
        save(invocation_path, invocation)
        calls = []
        for path in root.glob('*/*.json'):
            if path.name == 'initial.json' or path.name.startswith('counterfactual-'):
                data = json.loads(path.read_text())['generation']
                calls.append({'artifact': str(path), 'processor': 'qwen-generation', 'latency_seconds': data['latency_seconds'], 'usage': data['usage']})
            elif path.name.startswith('replay-'):
                data = json.loads(path.read_text())
                if data['model_calls']:
                    calls.append({'artifact': str(path), 'processor': 'd1' if path.stem == 'replay-B' else 'qwen-decision', 'latency_seconds': data['latency_seconds'], 'raw': data['raw']})
            elif path.name.startswith('arm-'):
                data = json.loads(path.read_text())
                for index, step in enumerate(data['attempts'][1:], 1):
                    # First bounded repair is already counted as a counterfactual.
                    if index == 1 and data['variant'] in ['B', 'C', 'D']:
                        continue
                    calls.append({'artifact': str(path), 'step': index, 'processor': 'qwen-generation', 'latency_seconds': step['generation']['latency_seconds'], 'usage': step['generation']['usage']})
                for index, step in enumerate(data['decisions'][1:], 2):
                    if step['model_calls']:
                        calls.append({'artifact': str(path), 'step': index, 'processor': 'd1' if data['variant'] == 'B' else 'qwen-decision', 'latency_seconds': step['latency_seconds'], 'raw': step['raw']})
        reused = sum(1 for p in root.glob('*/initial.json') if json.loads(p.read_text()).get('source_initial_artifact'))
        save(root / 'unique-calls.json', {'count': len(calls), 'reused_initial_generations': reused, 'new_completed_model_calls': len(calls) - reused, 'calls': calls, 'note': 'Counts completed persisted model calls, including replay/counterfactual overhead; failed HTTP calls are not represented. Reused initial generations are charged to logical arms but excluded from new completed calls.'})
    print('PILOT_COMPLETE', json.dumps(summarize(rows, initial, replays, config)), flush=True)

if __name__ == '__main__':
    main()
