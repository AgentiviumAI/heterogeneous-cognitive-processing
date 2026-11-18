import argparse
import importlib.metadata
import json
import time
from pathlib import Path
import yaml
from .clients import Clients
from .common import digest, file_hash, order_key, write_new
from experiments.monitor import Monitor
from benchmarks.bfcl.evaluator import correct

MODELS = ['qwen3-8b', 'd1-3b']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/affinity-v1.yaml')
    parser.add_argument('--mode', choices=['smoke', 'warmup', 'full'], required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    freeze_path = Path(config['manifest_root']) / 'freeze.json'
    freeze = json.loads(freeze_path.read_text())
    assert freeze['config_hash'] == digest(config)
    assert all(file_hash(p) == sha for p, sha in freeze['manifest_sha256'].items())
    root = Path(config['results'])
    root.mkdir(parents=True, exist_ok=True)
    source = {str(p): file_hash(p) for prefix in ['experiments/affinity', 'benchmarks/bfcl', 'benchmarks/scifact'] for p in Path(prefix).rglob('*.py')}
    for name in ['processors/qwen.py', 'processors/d1.py', 'experiments/monitor.py']:
        source[name] = file_hash(name)
    metadata_path = root / 'metadata.json'
    if not metadata_path.exists():
        write_new(metadata_path, {'config': config, 'config_hash': digest(config), 'source_sha256': source, 'freeze_sha256': file_hash(freeze_path), 'created_utc': time.time(), 'packages': {k: importlib.metadata.version(k) for k in ['httpx', 'pydantic', 'rank-bm25', 'numpy', 'nvidia-ml-py']}, 'protocol': 'sequential paired requests, counterbalanced processor order, 3 passes, same frozen semantic state, identical greedy seed across passes', 'unknowns': {'MIG_utilization': None, 'GPU_execution_time': None}})
    metadata = json.loads(metadata_path.read_text())
    assert metadata['source_sha256'] == source, 'Source changed: use a new experiment ID/directory.'
    clients = Clients(config)
    clients.health()
    if args.mode == 'full':
        assert (root / 'smoke/validation.json').exists(), 'Validate smoke answers before full measurement.'
        assert (root / 'warmup/completed.json').exists(), 'Warm up both processors first.'
    monitor = Monitor(); monitor.start()
    started = time.monotonic()
    try:
        pass_count = config['passes'] if args.mode == 'full' else 1
        for measurement_pass in range(pass_count):
            dataset_order = ['bfcl', 'scifact'] if measurement_pass % 2 == 0 else ['scifact', 'bfcl']
            for dataset in dataset_order:
                manifest = json.loads((Path(config['manifest_root']) / f'{dataset}.json').read_text())
                examples = manifest['examples']
                if args.mode != 'full':
                    # SciFact smoke covers 10 distinct queries, one candidate each.
                    examples = examples[:10] if dataset == 'bfcl' else [r for r in examples if r['retrieval_rank'] == 1][:10]
                examples = sorted(examples, key=lambda r: order_key(config['seed'] + measurement_pass, r['task_id']))
                for index, example in enumerate(examples):
                    operation = example['operation']
                    assert digest(operation) == example['input_state_hash']
                    model_order = MODELS if (index + measurement_pass) % 2 == 0 else MODELS[::-1]
                    for position, processor in enumerate(model_order):
                        filename = order_key(config['seed'], example['task_id']) + f'-{processor}.json'
                        path = root / args.mode / dataset / f'pass-{measurement_pass}' / filename
                        if path.exists():
                            row = json.loads(path.read_text())
                            assert row['input_state_hash'] == example['input_state_hash']
                            continue
                        result = clients.execute(processor, operation)
                        is_qwen = processor == 'qwen3-8b'
                        row = {'experiment_id': config['experiment_id'], 'dataset': dataset, 'task_id': example['task_id'], 'operation': 'tool_selection' if dataset == 'bfcl' else 'evidence_relevance', 'processor': processor, 'pass': measurement_pass, 'phase': args.mode, 'request_index_within_dataset': index * 2 + position, 'processor_order': position, 'input_state_hash': example['input_state_hash'], 'seed': config['seed'], 'model_revision': config['qwen_revision'] if is_qwen else config['d1_revision'], 'quantization': config['qwen_quantization'] if is_qwen else config['d1_quantization'], 'serving_backend': config['qwen_backend'] if is_qwen else config['d1_backend'], 'input_state': operation, **result}
                        if dataset == 'bfcl':
                            row['correct'] = correct(row['decision'], example['expected_decision'], row['valid'])
                            row['expected_decision'] = example['expected_decision']
                            row['category'] = example['category']
                        else:
                            row['query_id'], row['document_id'] = example['query_id'], example['document_id']
                            row['score'] = int(row['decision']) if row['valid'] else None
                        write_new(path, row)
                    if index % 25 == 0:
                        print('PROGRESS', args.mode, dataset, measurement_pass, index + 1, '/', len(examples), flush=True)
                print('PASS_DONE', args.mode, dataset, measurement_pass, flush=True)
    finally:
        rows = monitor.finish()
        observation = root / 'observations' / f'{args.mode}-{time.time_ns()}.json'
        write_new(observation, {'phase': args.mode, 'invocation_wall_seconds': time.monotonic() - started, 'gpu_memory_samples': rows, 'peak_vram_bytes': max((r['used_bytes'] for r in rows), default=None), 'GPU_time_seconds': None, 'sampling_interval_seconds': 0.5})
    if args.mode == 'smoke':
        from .validate import validate_smoke
        validate_smoke(root)
    if args.mode == 'warmup':
        if not (root / 'warmup/completed.json').exists():
            write_new(root / 'warmup/completed.json', {'per_processor_per_dataset': 10, 'excluded_from_analysis': True, 'completed_utc': time.time()})
    elif not (root / f'{args.mode}/completed.json').exists():
        write_new(root / f'{args.mode}/completed.json', {'passes': pass_count, 'completed_utc': time.time()})
    print('COMPLETED', args.mode, flush=True)

if __name__ == '__main__':
    main()
