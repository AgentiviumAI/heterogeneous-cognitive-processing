import ast
import json
from pathlib import Path
from experiments.affinity.common import bfcl_operation, digest, file_hash, order_key

def load_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate BFCL IDs: ' + str(path))
    return rows

def callable_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return callable_name(node.value) + '.' + node.attr
    raise ValueError('unsupported callable expression')

def expected_name(category, row, reference):
    if category in ['irrelevance', 'live_irrelevance']:
        return 'NO_TOOL', 'official irrelevance category semantics'
    ground = (reference or row).get('ground_truth')
    if not isinstance(ground, list) or len(ground) != 1:
        raise ValueError('not one reference call')
    if isinstance(ground[0], dict) and len(ground[0]) == 1:
        return next(iter(ground[0])), 'official possible_answer tool key'
    if isinstance(ground[0], str):
        expression = ast.parse(ground[0], mode='eval').body
        if isinstance(expression, ast.Call):
            return callable_name(expression.func), 'official executable ground_truth callable AST'
    raise ValueError('ambiguous/unsupported reference')

def prepare(raw, config, fits):
    selected, excluded, checksums, counts = [], [], {}, {}
    seen_states = set()
    for category, quota in config['bfcl_quotas'].items():
        data_path = Path(raw) / f'BFCL_v3_{category}.json'
        checksums[str(data_path)] = file_hash(data_path)
        answer_path = Path(raw) / 'possible_answer' / data_path.name
        references = {}
        if answer_path.exists():
            checksums[str(answer_path)] = file_hash(answer_path)
            references = {r['id']: r for r in load_jsonl(answer_path)}
        accepted = []
        for row in sorted(load_jsonl(data_path), key=lambda r: order_key(config['seed'], r['id'])):
            try:
                if len(row['question']) != 1 or not row['function']:
                    raise ValueError('not single-turn with candidates')
                names = [f['name'] for f in row['function']]
                if len(set(names)) != len(names) or len(names) > 16:
                    raise ValueError('duplicate names or >16 candidates')
                if category not in ['exec_simple', 'irrelevance', 'live_irrelevance'] and row['id'] not in references:
                    raise ValueError('missing corresponding reference ID')
                expected, origin = expected_name(category, row, references.get(row['id']))
                if expected != 'NO_TOOL' and expected not in names:
                    raise ValueError('reference name not in candidates')
                operation = bfcl_operation(row)
                operation_hash = digest(operation)
                if operation_hash in seen_states:
                    raise ValueError('duplicate frozen semantic operation')
                token_info = fits(operation)
                label = 'NO_TOOL' if expected == 'NO_TOOL' else f'T{names.index(expected)}'
                accepted.append({'task_id': row['id'], 'category': category, 'operation': operation, 'input_state_hash': digest(operation), 'expected_decision': label, 'expected_tool_name': expected, 'reference_origin': origin, 'token_preflight': token_info})
                seen_states.add(operation_hash)
            except (ValueError, KeyError, TypeError, SyntaxError) as error:
                excluded.append({'task_id': row['id'], 'category': category, 'reason': str(error)})
            if len(accepted) == quota:
                break
        if len(accepted) != quota:
            raise ValueError(f'{category}: only {len(accepted)} eligible examples, quota {quota}; do not silently resample a different category')
        counts[category] = len(accepted)
        selected.extend(accepted)
    assert len(selected) == config['bfcl_total']
    return {'dataset': 'BFCL V3 single-call tool-selection subset', 'revision': config['bfcl_revision'], 'source': 'gorilla-llm/Berkeley-Function-Calling-Leaderboard', 'seed': config['seed'], 'sampling': 'stratified quota; ascending SHA256(seed,id); eligibility checked before inference', 'file_sha256': checksums, 'category_counts': counts, 'excluded_before_selection': excluded, 'examples': selected}
