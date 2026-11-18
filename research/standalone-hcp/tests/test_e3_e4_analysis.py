import json
from pathlib import Path

from experiments.analyze_e3_e4 import analyze


def test_first_decision_counterfactual_uses_task_as_unit(tmp_path, monkeypatch):
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'mpl'))
    source = tmp_path / 'source'
    task_root = source / '11'
    (task_root).mkdir(parents=True)
    (source / 'dataset').mkdir()
    task = {'id': 11, 'prompt': 'Return one', 'tests': ['assert f() == 1'], 'setup': []}
    (source / 'dataset/tasks.json').write_text(json.dumps([task]))

    def generation(code, tokens=4):
        return {'code': code, 'latency_seconds': .1, 'usage': {'completion_tokens': tokens}, 'request': {'prompt': 'frozen request'}}

    def evaluation(passed):
        return {'passed': passed, 'feedback': 'HCP_ALL_TESTS_PASS' if passed else 'AssertionError', 'latency_seconds': .01, 'timed_out': False}

    initial = {'generation': generation('def f(): return 0'), 'evaluation': evaluation(False)}
    (task_root / 'initial.json').write_text(json.dumps(initial))
    (task_root / 'counterfactual-REPAIR.json').write_text(json.dumps({'generation': generation('def f(): return 1'), 'evaluation': evaluation(True)}))
    (task_root / 'counterfactual-REGENERATE.json').write_text(json.dumps({'generation': generation('def f(): return 0'), 'evaluation': evaluation(False)}))
    for arm, action in zip('ABCD', ['DIAGNOSE_AND_REPAIR', 'REPAIR', 'REGENERATE', 'REPAIR']):
        row = {'id': 11, 'variant': arm, 'initial_passed': False, 'passed': arm in 'BC', 'attempt_count': 2, 'attempts': [initial, {'generation': generation('def f(): return 1'), 'evaluation': evaluation(True)}], 'decisions': [{'action': action, 'latency_seconds': .02, 'raw': {'request': {'messages': []}}}], 'model_calls': 3, 'logical_workflow_seconds': .2}
        (task_root / f'arm-{arm}.json').write_text(json.dumps(row))
        (task_root / f'replay-{arm}.json').write_text(json.dumps({'action': action, 'latency_seconds': .02}))
    (source / 'unique-calls.json').write_text(json.dumps({'count': 1, 'reused_initial_generations': 0, 'new_completed_model_calls': 1, 'calls': [{'artifact': 'results/extended/11/initial.json', 'processor': 'qwen-generation', 'latency_seconds': .1, 'usage': {'prompt_tokens': 5, 'completion_tokens': 4}}]}))

    output = tmp_path / 'analysis'
    analyze(source, output, samples=100, seed=7)
    summary = json.loads((output / 'summary.json').read_text())
    record = json.loads((output / 'counterfactual-records.json').read_text())[0]
    assert summary['task_count'] == 1
    assert summary['counterfactual']['eligible_failed_states'] == 1
    assert summary['counterfactual']['decision_match_by_processor']['D1']['matched'] == 1
    assert summary['counterfactual']['decision_regret_by_processor']['D1'] == 0
    assert summary['actual_call_accounting']['new_completed_calls'] == 1
    assert summary['verification']['passed']
    assert len(record['failure_state_sha256']) == 64
    assert record['counterfactuals']['REPAIR']['evaluation']['passed']
    assert (output / 'figures/counterfactual-action-outcomes.pdf').exists()
