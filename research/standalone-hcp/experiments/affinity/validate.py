import json
from .common import write_new, digest

def validate_smoke(root):
    rows = [json.loads(p.read_text()) for p in (root / 'smoke').glob('*/pass-*/*.json')]
    assert len(rows) == 40, 'Smoke requires 10 examples × 2 datasets × 2 processors.'
    for row in rows:
        assert row['valid'], row['error']
        assert digest(row['input_state']) == row['input_state_hash']
        if row['dataset'] == 'scifact':
            assert row['score'] in [0, 1, 2, 3]
    grouped = {}
    for row in rows:
        grouped.setdefault((row['dataset'], row['task_id']), []).append(row)
    for key, pair in grouped.items():
        assert len(pair) == 2 and pair[0]['input_state_hash'] == pair[1]['input_state_hash'], key
    if not (root / 'smoke/validation.json').exists():
        write_new(root / 'smoke/validation.json', {'count': len(rows), 'valid': True, 'same_state_pairs': len(grouped), 'purpose': 'Extraction and shared contract validation only, not comparative prompt optimization.'})
