"""Task-clustered analysis for immutable E3 traces and E4 first-decision counterfactuals."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml


def bootstrap(values, rng, samples):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    return [float(x) for x in np.percentile(values[draws].mean(axis=1), [2.5, 97.5])]


def load_rows(root):
    rows = []
    for path in sorted(root.glob("*/arm-[ABCD].json")):
        rows.append(json.loads(path.read_text()))
    return rows


def analyze(source, output, samples, seed):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Result directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = load_rows(source)
    task_by_id = {int(t['id']): t for t in json.loads((source / 'dataset/tasks.json').read_text())}
    by_arm = {arm: {r['id']: r for r in rows if r['variant'] == arm} for arm in 'ABCD'}
    ids = sorted(set.intersection(*(set(v) for v in by_arm.values())))
    if not ids:
        raise ValueError("No complete four-policy paired records")
    summary = {'source_results': str(source), 'task_count': len(ids), 'bootstrap_unit': 'task id; one paired observation per task', 'success_definition': 'success within three workflow attempts; not statistical pass@3', 'policies': {}, 'paired': {}, 'failure_types': {}, 'counterfactual': {}}
    table = []
    for arm in 'ABCD':
        arm_rows = [by_arm[arm][i] for i in ids]
        outcomes = np.array([r['passed'] for r in arm_rows], dtype=float)
        initial_passes = np.array([r['initial_passed'] for r in arm_rows], dtype=float)
        condition = [r['passed'] for r in arm_rows if not r['initial_passed']]
        logical = np.array([r['logical_workflow_seconds'] for r in arm_rows], dtype=float)
        decisions = [d['latency_seconds'] for r in arm_rows for d in r['decisions']]
        calls = [r['model_calls'] for r in arm_rows]
        generated_tokens = 0
        for row in arm_rows:
            for attempt in row['attempts']:
                generated_tokens += int((attempt.get('generation', {}).get('usage') or {}).get('completion_tokens', 0) or 0)
            for decision in row['decisions']:
                raw = decision.get('raw') or {}
                if isinstance(raw, dict):
                    generated_tokens += int((raw.get('usage') or raw.get('response', {}).get('usage') or {}).get('completion_tokens', 0) or 0)
        successes = int(outcomes.sum())
        ci = bootstrap(outcomes, rng, samples)
        summary['policies'][arm] = {'n': len(ids), 'initial_passes': int(initial_passes.sum()), 'successes': successes, 'success_rate': successes / len(ids), 'success_ci95_task_bootstrap': ci, 'conditional_success_after_initial_failure': float(np.mean(condition)) if condition else None, 'logical_latency_mean_seconds': float(logical.mean()), 'logical_latency_median_seconds': float(np.median(logical)), 'decision_latency_mean_seconds': float(np.mean(decisions)) if decisions else 0.0, 'model_calls_mean': float(np.mean(calls)), 'logical_generated_tokens': generated_tokens, 'repair_attempts_mean': float(np.mean([max(0, r['attempt_count'] - 1) for r in arm_rows])), 'evaluator_timeouts': sum(bool(attempt.get('evaluation', {}).get('timed_out')) for r in arm_rows for attempt in r['attempts']), 'counterfactual_reuse': 'initial bounded-action generations may be shared across policies; logical latency is reconstructed and not physical per-policy wall time'}
        table.append({'policy': arm, 'n': len(ids), 'initial_passes': int(initial_passes.sum()), 'successes': successes, 'success_rate': successes / len(ids), 'ci95_low': ci[0], 'ci95_high': ci[1], 'conditional_success_after_initial_failure': summary['policies'][arm]['conditional_success_after_initial_failure'], 'logical_latency_median_seconds': float(np.median(logical)), 'decision_latency_mean_seconds': summary['policies'][arm]['decision_latency_mean_seconds'], 'model_calls_mean': float(np.mean(calls)), 'logical_generated_tokens': generated_tokens, 'repair_attempts_mean': summary['policies'][arm]['repair_attempts_mean'], 'evaluator_timeouts': summary['policies'][arm]['evaluator_timeouts']})
    for left, right in itertools.combinations('ABCD', 2):
        left_outcome = np.array([by_arm[left][i]['passed'] for i in ids], dtype=bool)
        right_outcome = np.array([by_arm[right][i]['passed'] for i in ids], dtype=bool)
        diff = right_outcome.astype(float) - left_outcome.astype(float)
        counts = {'wins': int(np.sum(right_outcome & ~left_outcome)), 'losses': int(np.sum(~right_outcome & left_outcome)), 'ties': int(np.sum(right_outcome == left_outcome))}
        counts['success_rate_difference'] = float(diff.mean())
        counts['difference_ci95_task_bootstrap'] = bootstrap(diff, rng, samples)
        summary['paired'][f'{right}_vs_{left}'] = counts
    for r in rows:
        if r['variant'] != 'A':
            continue
        task_dir = source / str(r['id'])
        initial = json.loads((task_dir / 'initial.json').read_text())
        if initial['evaluation']['passed']:
            continue
        typ = 'timeout' if initial['evaluation'].get('timed_out') else 'other_failure'
        summary['failure_types'].setdefault(typ, {'tasks': 0, 'policy_success': {a: 0 for a in 'ABCD'}})
        summary['failure_types'][typ]['tasks'] += 1
        for arm in 'ABCD':
            summary['failure_types'][typ]['policy_success'][arm] += int(by_arm[arm][r['id']]['passed'])
    counterfactual_rows = []
    decisions_by_policy = {a: {} for a in 'ABCD'}
    for path in sorted(source.glob('*/replay-[ABCD].json')):
        task_id = int(path.parent.name)
        record = json.loads(path.read_text())
        decisions_by_policy[path.stem[-1]][task_id] = record
    states = []
    for task_id in sorted(set.intersection(*(set(v) for v in decisions_by_policy.values())) if all(decisions_by_policy[a] for a in 'ABCD') else set()):
        task_dir = source / str(task_id)
        initial = json.loads((task_dir / 'initial.json').read_text())
        repair = json.loads((task_dir / 'counterfactual-REPAIR.json').read_text())
        regenerate = json.loads((task_dir / 'counterfactual-REGENERATE.json').read_text())
        outcomes = {'REPAIR': bool(repair['evaluation']['passed']), 'REGENERATE': bool(regenerate['evaluation']['passed'])}
        better = [a for a, passed in outcomes.items() if passed == max(outcomes.values())]
        choices = {a: decisions_by_policy[a][task_id]['action'] for a in 'ABCD'}
        task = task_by_id[task_id]
        failure_state = {'requirement': task['prompt'], 'code': initial['generation']['code'], 'test_feedback': initial['evaluation']['feedback'][-2000:]}
        state_hash = hashlib.sha256(json.dumps(failure_state, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        row = {'task_id': task_id, 'failure_state': failure_state, 'failure_state_sha256': state_hash, 'action_outcomes': outcomes, 'counterfactuals': {'REPAIR': {'generation_request': repair['generation'].get('request'), 'generated_code': repair['generation']['code'], 'evaluation': repair['evaluation'], 'generation_latency_seconds': repair['generation']['latency_seconds'], 'usage': repair['generation']['usage']}, 'REGENERATE': {'generation_request': regenerate['generation'].get('request'), 'generated_code': regenerate['generation']['code'], 'evaluation': regenerate['evaluation'], 'generation_latency_seconds': regenerate['generation']['latency_seconds'], 'usage': regenerate['generation']['usage']}}, 'better_observed_actions': better, 'tie': len(better) == 2, 'selected_actions': {**choices, 'D1': choices['B'], 'Qwen': choices['D'], 'deterministic_rule': choices['C']}, 'decision_latency_seconds': {a: decisions_by_policy[a][task_id]['latency_seconds'] for a in 'ABCD'}, 'action_generation_latency_seconds': {'REPAIR': repair['generation']['latency_seconds'], 'REGENERATE': regenerate['generation']['latency_seconds']}, 'action_generation_tokens': {'REPAIR': repair['generation']['usage'], 'REGENERATE': regenerate['generation']['usage']}}
        counterfactual_rows.append(row)
        states.append(row)
    actions_n = len(counterfactual_rows)
    expected_ids = {int(t['id']) for t in task_by_id.values()}
    failed_ids = {task_id for task_id in expected_ids if (source / str(task_id) / 'initial.json').exists() and not json.loads((source / str(task_id) / 'initial.json').read_text())['evaluation']['passed']}
    paired_ids = set(ids)
    counterfactual_ids = {row['task_id'] for row in counterfactual_rows}
    missing_counterfactuals = sorted(failed_ids - counterfactual_ids)
    unexpected_counterfactuals = sorted(counterfactual_ids - failed_ids)
    dataset_metadata_path = source / 'dataset/dataset.json'
    dataset_metadata = json.loads(dataset_metadata_path.read_text()) if dataset_metadata_path.exists() else None
    dataset_hash = hashlib.sha256((source / 'dataset/tasks.json').read_bytes()).hexdigest()
    checks = {'all_dataset_tasks_have_all_four_policy_records': paired_ids == expected_ids, 'all_initial_failures_have_four_decisions_and_two_counterfactual_actions': not missing_counterfactuals and not unexpected_counterfactuals and len(counterfactual_ids) == len(failed_ids), 'dataset_file_matches_dataset_manifest': dataset_metadata is None or dataset_metadata.get('sha256') == dataset_hash, 'failure_state_hashes_present': all(len(row['failure_state_sha256']) == 64 for row in counterfactual_rows), 'counterfactual_request_and_evaluation_traces_present': all(all(row['counterfactuals'][action]['generation_request'] and row['counterfactuals'][action]['evaluation'] for action in ['REPAIR', 'REGENERATE']) for row in counterfactual_rows), 'generation_and_model_decision_request_traces_present': all(all(a.get('generation', {}).get('request') for a in arm['attempts']) and all(d.get('raw', {}).get('request') for d in arm['decisions'] if arm['variant'] in 'ABD') for arm in rows)}
    audit = {'passed': all(checks.values()), 'checks': checks, 'dataset_tasks': len(expected_ids), 'paired_policy_tasks': len(paired_ids), 'initial_failed_tasks': len(failed_ids), 'counterfactual_states': len(counterfactual_ids), 'missing_counterfactual_task_ids': missing_counterfactuals, 'unexpected_counterfactual_task_ids': unexpected_counterfactuals, 'dataset_sha256': dataset_hash}
    (output / 'verification-audit.json').write_text(json.dumps(audit, indent=2))
    if not audit['passed']:
        raise ValueError(f"Trace integrity audit failed: {audit}")
    summary['verification'] = audit
    incidents_path = source / 'transport-incidents.json'
    transport_incidents = json.loads(incidents_path.read_text()) if incidents_path.exists() else []
    summary['transport_incidents'] = {'unresolved_attempts': len(transport_incidents), 'records': transport_incidents, 'included_in_completed_unique_call_count': False, 'accounting_note': 'No response was observed; backend completion is unknown.'}
    action_matrix = {a: {str(b): sum(int(x['action_outcomes'][a]) == int(b) for x in counterfactual_rows) for b in [0, 1]} for a in ['REPAIR', 'REGENERATE']}
    repair = np.asarray([x['action_outcomes']['REPAIR'] for x in counterfactual_rows], dtype=float)
    regenerate = np.asarray([x['action_outcomes']['REGENERATE'] for x in counterfactual_rows], dtype=float)
    different = np.asarray([x['action_outcomes']['REPAIR'] != x['action_outcomes']['REGENERATE'] for x in counterfactual_rows], dtype=float)
    equivalent = 1 - different
    summary['counterfactual'] = {'eligible_failed_states': actions_n, 'action_outcome_matrix': action_matrix, 'repair_pass_rate': float(repair.mean()) if actions_n else None, 'repair_pass_rate_ci95_task_bootstrap': bootstrap(repair, rng, samples), 'regenerate_pass_rate': float(regenerate.mean()) if actions_n else None, 'regenerate_pass_rate_ci95_task_bootstrap': bootstrap(regenerate, rng, samples), 'repair_minus_regenerate_success_difference': float((repair-regenerate).mean()) if actions_n else None, 'repair_minus_regenerate_ci95_task_bootstrap': bootstrap(repair-regenerate, rng, samples), 'different_outcomes': int(different.sum()), 'different_outcome_rate': float(different.mean()) if actions_n else None, 'different_outcome_rate_ci95_task_bootstrap': bootstrap(different, rng, samples), 'equivalent_outcomes': int(equivalent.sum()), 'equivalent_outcome_rate': float(equivalent.mean()) if actions_n else None, 'equivalent_outcome_rate_ci95_task_bootstrap': bootstrap(equivalent, rng, samples), 'decision_match_and_better_action': {}, 'hindsight_oracle_success_rate': float(np.mean([max(x['action_outcomes'].values()) for x in counterfactual_rows])) if actions_n else None, 'first_decision_only': True, 'oracle_definition': 'best observed outcome among the two generated first-action counterfactuals; ties count as matched for either action and this is not a deployable policy'}
    for arm in 'ABCD':
        matched = np.asarray([x['selected_actions'][arm] in x['better_observed_actions'] for x in counterfactual_rows], dtype=float) if arm in 'BCD' else np.asarray([])
        summary['counterfactual']['decision_match_and_better_action'][arm] = {'matched': int(matched.sum()), 'eligible': len(matched), 'rate': float(matched.mean()) if len(matched) else None, 'ci95_task_bootstrap': bootstrap(matched, rng, samples) if len(matched) else [None, None], 'not_applicable': arm == 'A'}
    summary['counterfactual']['decision_regret'] = {arm: (float(np.mean([int(max(x['action_outcomes'].values())) - int(x['action_outcomes'][x['selected_actions'][arm]]) for x in counterfactual_rows])) if actions_n else None) if arm in 'BCD' else None for arm in 'ABCD'}
    summary['counterfactual']['decision_regret_ci95_task_bootstrap'] = {arm: bootstrap(np.asarray([int(max(x['action_outcomes'].values())) - int(x['action_outcomes'][x['selected_actions'][arm]]) for x in counterfactual_rows], dtype=float), rng, samples) if actions_n and arm in 'BCD' else [None, None] for arm in 'ABCD'}
    summary['counterfactual']['processor_policy_mapping'] = {'D1': 'B', 'Qwen_bounded': 'D', 'deterministic_rule': 'C'}
    summary['counterfactual']['decision_match_by_processor'] = {'D1': summary['counterfactual']['decision_match_and_better_action']['B'], 'Qwen_bounded': summary['counterfactual']['decision_match_and_better_action']['D'], 'deterministic_rule': summary['counterfactual']['decision_match_and_better_action']['C']}
    summary['counterfactual']['decision_regret_by_processor'] = {'D1': summary['counterfactual']['decision_regret']['B'], 'Qwen_bounded': summary['counterfactual']['decision_regret']['D'], 'deterministic_rule': summary['counterfactual']['decision_regret']['C']}
    failure_group = {}
    for row in counterfactual_rows:
        evaluation = json.loads((source / str(row['task_id']) / 'initial.json').read_text())['evaluation']
        feedback = evaluation.get('feedback', '')
        error = next((name for name in ['AssertionError', 'SyntaxError', 'NameError', 'TypeError', 'ValueError', 'IndexError', 'KeyError'] if name in feedback), 'timeout' if evaluation.get('timed_out') else 'other_failure')
        task = task_by_id[row['task_id']]
        difficulty = len(task['prompt']) + sum(len(test) for test in task['tests']) + sum(len(str(item)) for item in task.get('setup', []))
        failure_group.setdefault(error, []).append((row, difficulty))
    summary['counterfactual']['by_failure_type'] = {}
    for failure, items in failure_group.items():
        summary['counterfactual']['by_failure_type'][failure] = {'states': len(items), 'repair_passes': sum(x['action_outcomes']['REPAIR'] for x, _ in items), 'regenerate_passes': sum(x['action_outcomes']['REGENERATE'] for x, _ in items), 'different_action_outcomes': sum(x['action_outcomes']['REPAIR'] != x['action_outcomes']['REGENERATE'] for x, _ in items), 'decision_match_rate': {arm: (float(np.mean([x['selected_actions'][arm] in x['better_observed_actions'] for x, _ in items])) if arm in 'BCD' else None) for arm in 'ABCD'}}
    difficulty_by_id = {task_id: len(task['prompt']) + sum(len(test) for test in task['tests']) + sum(len(str(item)) for item in task.get('setup', [])) for task_id, task in task_by_id.items()}
    ordered_difficulty = sorted(((difficulty_by_id[row['task_id']], row) for row in counterfactual_rows), key=lambda item: item[0])
    difficulty_groups = {'lower': [], 'middle': [], 'upper': []}
    for index, (_, row) in enumerate(ordered_difficulty):
        group = 'lower' if index < len(ordered_difficulty) / 3 else 'middle' if index < 2 * len(ordered_difficulty) / 3 else 'upper'
        difficulty_groups[group].append(row)
    summary['counterfactual']['by_task_difficulty'] = {group: {'states': len(items), 'repair_passes': sum(x['action_outcomes']['REPAIR'] for x in items), 'regenerate_passes': sum(x['action_outcomes']['REGENERATE'] for x in items), 'decision_match_rate': {arm: (float(np.mean([x['selected_actions'][arm] in x['better_observed_actions'] for x in items])) if items else None) if arm in 'BCD' else None for arm in 'ABCD'}} for group, items in difficulty_groups.items()}
    downstream = []
    for row in counterfactual_rows:
        for arm in 'BCD':
            downstream.append({'policy': arm, 'decision_latency_seconds': row['decision_latency_seconds'][arm], 'selected_action_passed': row['action_outcomes'][row['selected_actions'][arm]], 'selected_action_generation_latency_seconds': row['action_generation_latency_seconds'][row['selected_actions'][arm]], 'downstream_calls': 1, 'downstream_tokens': row['action_generation_tokens'][row['selected_actions'][arm]]})
    summary['counterfactual']['decision_latency_vs_downstream'] = {'samples': len(downstream), 'by_policy': {arm: {'mean_decision_latency_seconds': float(np.mean([x['decision_latency_seconds'] for x in downstream if x['policy'] == arm])), 'selected_action_pass_rate': float(np.mean([x['selected_action_passed'] for x in downstream if x['policy'] == arm])), 'mean_generation_latency_seconds': float(np.mean([x['selected_action_generation_latency_seconds'] for x in downstream if x['policy'] == arm]))} for arm in 'BCD'}, 'free_form_baseline_A': 'Excluded from REPAIR/REGENERATE counterfactual action-match analysis because it emits diagnosis text, not one bounded action.', 'downstream_cost_can_exceed_decision_savings': 'Compare policy decision latency and selected-action generation latency; generation call count is one per observed first-action branch.'}
    summary['counterfactual']['decision_latency_vs_downstream']['by_processor'] = {'D1': summary['counterfactual']['decision_latency_vs_downstream']['by_policy']['B'], 'Qwen_bounded': summary['counterfactual']['decision_latency_vs_downstream']['by_policy']['D'], 'deterministic_rule': summary['counterfactual']['decision_latency_vs_downstream']['by_policy']['C']}
    invocations = [json.loads(path.read_text()) for path in sorted((source / 'run-invocations').glob('*.json'))] if (source / 'run-invocations').exists() else []
    summary['measured_suite_wall_time'] = {'invocations': len(invocations), 'total_seconds_across_resumptions': sum(x['suite_wall_seconds_this_invocation'] for x in invocations), 'per_invocation_seconds': [x['suite_wall_seconds_this_invocation'] for x in invocations], 'per_policy_actual_wall_time': None, 'note': 'Policies share initial generations and some counterfactual calls; only complete suite invocation wall time is measured, not independent physical wall time per policy.'}
    call_ledger = json.loads((source / 'unique-calls.json').read_text())
    call_totals = {}
    for item in call_ledger['calls']:
        processor = item['processor']
        call_totals.setdefault(processor, {'new_completed_calls': 0, 'reused_initial_generations': 0, 'reported_input_tokens': 0, 'reported_output_tokens': 0, 'reused_initial_input_tokens': 0, 'reused_initial_output_tokens': 0})
        is_reused = False
        if processor == 'qwen-generation' and item['artifact'].endswith('/initial.json'):
            artifact = source / item['artifact'].split('/')[-2] / item['artifact'].split('/')[-1]
            is_reused = bool(json.loads(artifact.read_text()).get('source_initial_artifact'))
        if is_reused:
            call_totals[processor]['reused_initial_generations'] += 1
        else:
            call_totals[processor]['new_completed_calls'] += 1
        raw = item.get('raw') or {}
        if processor == 'qwen-generation':
            usage = item.get('usage') or {}
        elif isinstance(raw, dict):
            usage = raw.get('usage') or raw.get('response', {}).get('usage') or {}
        else:
            usage = {}
        input_tokens = int(usage.get('prompt_tokens', usage.get('input_tokens', 0)) or 0)
        output_tokens = int(usage.get('completion_tokens', usage.get('output_tokens', 0)) or 0)
        if is_reused:
            call_totals[processor]['reused_initial_input_tokens'] += input_tokens
            call_totals[processor]['reused_initial_output_tokens'] += output_tokens
        else:
            call_totals[processor]['reported_input_tokens'] += input_tokens
            call_totals[processor]['reported_output_tokens'] += output_tokens
    summary['actual_call_accounting'] = {'new_completed_calls': sum(x['new_completed_calls'] for x in call_totals.values()), 'completed_call_artifacts_including_reused_initials': call_ledger['count'], 'reused_initial_generations': call_ledger['reused_initial_generations'], 'by_processor': call_totals, 'unresolved_transport_attempts': summary['transport_incidents']['unresolved_attempts'], 'accounting_note': 'One unresolved transport attempt had no response and is excluded. D1 native API reports zero output tokens for bounded-choice responses; this is not text generation.'}
    downstream_by_processor = summary['counterfactual']['decision_latency_vs_downstream']['by_processor']
    for name, item in downstream_by_processor.items():
        item['mean_decision_plus_selected_generation_seconds'] = item['mean_decision_latency_seconds'] + item['mean_generation_latency_seconds']
    (output / 'actual-call-accounting.csv').write_text('processor,new_completed_calls,reused_initial_generations,new_call_input_tokens,new_call_output_tokens,reused_initial_input_tokens,reused_initial_output_tokens\n' + ''.join(f"{processor},{values['new_completed_calls']},{values['reused_initial_generations']},{values['reported_input_tokens']},{values['reported_output_tokens']},{values['reused_initial_input_tokens']},{values['reused_initial_output_tokens']}\n" for processor, values in sorted(call_totals.items())))
    with (output / 'decision-latency-downstream.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'policy', 'decision_latency_seconds', 'selected_action', 'selected_action_passed', 'generation_latency_seconds', 'generation_tokens']); writer.writeheader()
        for row in counterfactual_rows:
            for arm in 'BCD':
                action = row['selected_actions'][arm]
                writer.writerow({'task_id': row['task_id'], 'policy': arm, 'decision_latency_seconds': row['decision_latency_seconds'][arm], 'selected_action': action, 'selected_action_passed': row['action_outcomes'][action], 'generation_latency_seconds': row['action_generation_latency_seconds'][action], 'generation_tokens': json.dumps(row['action_generation_tokens'][action])})
    (output / 'counterfactual-records.json').write_text(json.dumps(counterfactual_rows, indent=2))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (output / 'policy-comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys()); writer.writeheader(); writer.writerows(table)
    with (output / 'paired-policy-comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['comparison', 'wins', 'losses', 'ties', 'success_rate_difference', 'ci95_low', 'ci95_high']); writer.writeheader()
        for name, item in summary['paired'].items(): writer.writerow({'comparison': name, 'wins': item['wins'], 'losses': item['losses'], 'ties': item['ties'], 'success_rate_difference': item['success_rate_difference'], 'ci95_low': item['difference_ci95_task_bootstrap'][0], 'ci95_high': item['difference_ci95_task_bootstrap'][1]})
    with (output / 'workflow-latency-distribution.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'policy', 'initial_passed', 'workflow_passed', 'logical_workflow_seconds']); writer.writeheader()
        for arm in 'ABCD':
            for task_id in ids:
                row = by_arm[arm][task_id]
                writer.writerow({'task_id': task_id, 'policy': arm, 'initial_passed': row['initial_passed'], 'workflow_passed': row['passed'], 'logical_workflow_seconds': row['logical_workflow_seconds']})
    with (output / 'decision-latency-distribution.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'policy', 'decision_index', 'decision_latency_seconds']); writer.writeheader()
        for arm in 'ABCD':
            for task_id in ids:
                for index, decision in enumerate(by_arm[arm][task_id]['decisions'], 1):
                    writer.writerow({'task_id': task_id, 'policy': arm, 'decision_index': index, 'decision_latency_seconds': decision['latency_seconds']})
    (output / 'E3-E4-EXPERIMENT-REPORT.md').write_text(make_report(summary))
    try:
        figures(summary, table, counterfactual_rows, output)
    except ImportError:
        summary['figure_status'] = 'matplotlib unavailable; numerical outputs were produced'
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'tasks': len(ids), 'counterfactual_states': actions_n, 'output': str(output)}))
    return summary


def make_report(s):
    lines = [
        '# E3/E4 MBPP Experimental Report', '',
        '## E3: Extended MBPP workflow', '',
        'Success is passing within the three-attempt workflow and is not statistical pass@3. Confidence intervals use task-level paired bootstrap resampling. Shared initial and counterfactual generations mean per-policy latency is a logical reconstruction, not independent physical wall time. Latency distributions are in `workflow-latency-distribution.csv` and `decision-latency-distribution.csv`.', '',
        '| Policy | Success | Task-bootstrap 95% CI | Conditional after initial failure | Median logical latency (s) | Mean calls | Logical generated tokens |',
        '|---|---:|---:|---:|---:|---:|---:|'
    ]
    for arm, result in s['policies'].items():
        ci = result['success_ci95_task_bootstrap']
        conditional = result['conditional_success_after_initial_failure']
        conditional_text = 'n/a' if conditional is None else f'{conditional:.1%}'
        lines.append(f"| {arm} | {result['successes']}/{result['n']} ({result['success_rate']:.1%}) | [{ci[0]:.1%}, {ci[1]:.1%}] | {conditional_text} | {result['logical_latency_median_seconds']:.3f} | {result['model_calls_mean']:.2f} | {result['logical_generated_tokens']} |")
    initial = s['policies']['A']['initial_passes']
    lines.extend([
        '', f"Initial solution passed {initial}/{s['task_count']} tasks ({initial / s['task_count']:.1%}).",
        f"Measured suite wall time across {s['measured_suite_wall_time']['invocations']} invocations was {s['measured_suite_wall_time']['total_seconds_across_resumptions']:.1f}s. Per-policy physical wall time was not measured. All six policy comparisons are in `paired-policy-comparison.csv`; intervals are exploratory.",
        f"The suite persisted {s['actual_call_accounting']['new_completed_calls']} new completed model calls across policies (plus {s['actual_call_accounting']['reused_initial_generations']} initial generations reused from the pilot). Backend token accounting is in `actual-call-accounting.csv`; D1's bounded-choice API reports zero output tokens.",
        f"Unresolved transport attempts: {s['transport_incidents']['unresolved_attempts']}. Their request bodies are reconstructed, no response was observed, backend completion is unknown, and they are excluded from completed call counts.",
        '', 'GPU utilization and GPU execution time were unavailable on the local runner. Trace completeness and pairing checks are in `verification-audit.json`.', '',
        '### Figures', '', 'Success intervals are task-bootstrap intervals. Latency and outcome relationships are descriptive and do not identify causal effects.', '',
        '## E4: First-decision counterfactuals', ''
    ])
    cf = s['counterfactual']
    diff_ci = cf['different_outcome_rate_ci95_task_bootstrap']
    action_ci = cf['repair_minus_regenerate_ci95_task_bootstrap']
    lines.append(f"Eligible failed states: {cf['eligible_failed_states']}. REPAIR and REGENERATE differed in outcome for {cf['different_outcomes']}/{cf['eligible_failed_states']} ({cf['different_outcome_rate']:.1%}, 95% CI [{diff_ci[0]:.1%}, {diff_ci[1]:.1%}]). The paired REPAIR-minus-REGENERATE pass-rate difference was {cf['repair_minus_regenerate_success_difference']:+.1%} (95% CI [{action_ci[0]:+.1%}, {action_ci[1]:+.1%}]).")
    for name in ['D1', 'Qwen_bounded', 'deterministic_rule']:
        metric = cf['decision_match_by_processor'][name]
        interval = metric['ci95_task_bootstrap']
        lines.append(f"{name} matched the better observed action on {metric['matched']}/{metric['eligible']} states ({metric['rate']:.1%}, 95% CI [{interval[0]:.1%}, {interval[1]:.1%}]).")
    d1_cost = cf['decision_latency_vs_downstream']['by_processor']['D1']
    qwen_cost = cf['decision_latency_vs_downstream']['by_processor']['Qwen_bounded']
    lines.append(f"D1 mean decision latency was {d1_cost['mean_decision_latency_seconds']:.3f}s versus bounded Qwen {qwen_cost['mean_decision_latency_seconds']:.3f}s; selected-action generation latency was {d1_cost['mean_generation_latency_seconds']:.3f}s versus {qwen_cost['mean_generation_latency_seconds']:.3f}s. Mean first-decision-plus-generation logical latency was {d1_cost['mean_decision_plus_selected_generation_seconds']:.3f}s for D1 and {qwen_cost['mean_decision_plus_selected_generation_seconds']:.3f}s for bounded Qwen. This is a branch-level descriptive comparison, not a full-workflow causal estimate.")
    lines.extend([
        '', 'Action ties count as matches for either action; no gold label is forced. Failure-type and task-difficulty groups, downstream latency, and regret are in `summary.json` and `decision-latency-downstream.csv`.',
        '', 'The hindsight oracle is an analytical upper bound using observed first-action outcomes. It is not deployable and does not establish full-workflow optimality.',
        '', 'Decision speed alone does not establish end-to-end benefit. Model size, quantization, prompt/protocol, and backend vary, so causal specialization is not identified.',
        '', '## Reproduction', '', '`python -m experiments.analyze_e3_e4 --config configs/counterfactual-v1.yaml`', ''
    ])
    return '\n'.join(lines)

def figures(summary, table, cf, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = output / 'figures'; figures.mkdir(exist_ok=True)
    arms = list(summary['policies'])
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    values = [summary['policies'][a]['success_rate'] for a in arms]
    low = [values[i] - summary['policies'][a]['success_ci95_task_bootstrap'][0] for i, a in enumerate(arms)]
    high = [summary['policies'][a]['success_ci95_task_bootstrap'][1] - values[i] for i, a in enumerate(arms)]
    ax.bar(arms, values, color=['#276FBF','#F4A259','#43AA8B','#9B5DE5']); ax.errorbar(arms, values, yerr=[low, high], fmt='none', ecolor='black', capsize=4); ax.set_ylim(0, 1.05); ax.set_ylabel('Success within three attempts'); ax.set_xlabel('Policy'); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.savefig(figures / 'success-rate-ci.pdf'); fig.savefig(figures / 'success-rate-ci.png', dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    for a in arms: ax.scatter(summary['policies'][a]['decision_latency_mean_seconds'], summary['policies'][a]['success_rate'], label=a, s=65)
    ax.set_xlabel('Mean decision latency (s)'); ax.set_ylabel('Workflow success rate'); ax.set_ylim(0, 1.05); ax.legend(title='Policy'); ax.grid(alpha=.2)
    fig.savefig(figures / 'decision-latency-vs-outcome.pdf'); fig.savefig(figures / 'decision-latency-vs-outcome.png', dpi=300); plt.close(fig)
    workflow = {a: [] for a in arms}
    with (output / 'workflow-latency-distribution.csv').open() as handle:
        for row in csv.DictReader(handle): workflow[row['policy']].append(float(row['logical_workflow_seconds']))
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    ax.boxplot([workflow[a] for a in arms], tick_labels=arms, showfliers=False, patch_artist=True, boxprops={'facecolor':'#9BBCE0'}, medianprops={'color':'#19324D'})
    ax.set_xlabel('Policy'); ax.set_ylabel('Logical workflow latency (s)'); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.savefig(figures / 'workflow-latency-distribution.pdf'); fig.savefig(figures / 'workflow-latency-distribution.png', dpi=300); plt.close(fig)
    if cf:
        matrix = np.array([[sum(int(x['action_outcomes'][a]) == result for x in cf) for result in [0,1]] for a in ['REPAIR','REGENERATE']])
        fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True); ax.imshow(matrix, cmap='Blues'); ax.set_xticks([0,1], labels=['Fail','Pass']); ax.set_yticks([0,1], labels=['REPAIR','REGENERATE']); ax.set_xlabel('Observed candidate outcome')
        for y in range(2):
            for x in range(2): ax.text(x,y,str(matrix[y,x]),ha='center',va='center')
        fig.savefig(figures / 'counterfactual-action-outcomes.pdf'); fig.savefig(figures / 'counterfactual-action-outcomes.png', dpi=300); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
        colors = {'B':'#F4A259','C':'#43AA8B','D':'#9B5DE5'}
        for arm in 'BCD':
            xs = [row['decision_latency_seconds'][arm] for row in cf]
            ys = [int(row['action_outcomes'][row['selected_actions'][arm]]) + (((row['task_id'] % 13) - 6) / 60) for row in cf]
            ax.scatter(xs, ys, label={'B':'D1','C':'Fixed rule','D':'Qwen bounded'}[arm], color=colors[arm], alpha=.7, s=24)
        ax.set_xlabel('Decision latency (s)'); ax.set_ylabel('Selected counterfactual outcome (pass=1, fail=0)'); ax.set_yticks([0,1], labels=['Fail','Pass']); ax.legend(); ax.grid(axis='y', alpha=.2)
        fig.savefig(figures / 'decision-latency-downstream.pdf'); fig.savefig(figures / 'decision-latency-downstream.png', dpi=300); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--source')
    parser.add_argument('--output')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    output = Path(args.output or config.get('results', 'results/mbpp-extended-v1/analysis'))
    source = Path(args.source or config.get('source_results', 'results/mbpp-extended-v1'))
    samples = args.samples or config.get('bootstrap_samples', 10000)
    seed = args.seed or config.get('seed', 20261015)
    analyze(source, output, samples, seed)
    analysis_config = Path(args.config) if args.config else None
    try: commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception: commit = None
    experiment_metadata = json.loads((source / 'metadata.json').read_text())
    provenance = {'experiment_id': experiment_metadata['config'].get('experiment_id'), 'experiment_source_commit_sha': experiment_metadata.get('source_commit_sha'), 'analysis_source_commit_sha': commit, 'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'analysis_config': config, 'analysis_config_sha256': hashlib.sha256(analysis_config.read_bytes()).hexdigest() if analysis_config else None, 'bootstrap_samples': samples, 'seed': seed, 'dataset_sha256': hashlib.sha256((source / 'dataset/tasks.json').read_bytes()).hexdigest(), 'verification_audit_sha256': hashlib.sha256((output / 'verification-audit.json').read_bytes()).hexdigest()}
    (output / 'analysis-provenance.json').write_text(json.dumps(provenance, indent=2))
    report = (output / 'E3-E4-EXPERIMENT-REPORT.md').read_text()
    docs_path = Path(__file__).resolve().parents[3] / 'docs' / 'E3-E4-EXPERIMENT-REPORT.md'
    docs_path.write_text(report)


if __name__ == '__main__':
    main()
