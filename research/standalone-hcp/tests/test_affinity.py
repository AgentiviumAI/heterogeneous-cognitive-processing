import math
from benchmarks.bfcl.adapter import expected_name
from benchmarks.bfcl.evaluator import correct
from benchmarks.scifact.evaluator import ranking_metrics
from experiments.affinity.common import bfcl_operation, digest, scifact_operation
from experiments.affinity.analysis import bootstrap_pair

def test_reference_ast_and_no_call():
    row = {'ground_truth': ['tool.get(x=1)']}
    assert expected_name('exec_simple', row, None)[0] == 'tool.get'
    assert expected_name('multiple', {}, {'ground_truth': [{'a.b': {'x': [1]}}]})[0] == 'a.b'
    assert expected_name('irrelevance', {}, None)[0] == 'NO_TOOL'
    assert not correct(None, 'NO_TOOL', False)

def test_prompt_does_not_depend_on_gold():
    row = {'question': [[{'role': 'user', 'content': 'request'}]], 'function': [{'name': 'tool', 'description': 'desc', 'parameters': {}}], 'ground_truth': ['secret']}
    operation = bfcl_operation(row)
    row['ground_truth'] = ['different secret']
    assert digest(operation) == digest(bfcl_operation(row))
    assert 'ground_truth' not in str(operation)

def test_ranking_idcg_and_missing_relevant():
    assert ranking_metrics(['gold', 'unknown'], {'gold': 1})['ndcg10'] == 1
    result = ranking_metrics(['unknown', 'gold'], {'gold': 1}, pool=['unknown', 'gold'])
    assert math.isclose(result['ndcg10'], 1 / math.log2(3))
    assert result['mrr10'] == .5
    missing = ranking_metrics(['unknown'], {'gold': 1}, pool=['unknown'])
    assert missing['ndcg10'] == 0 and missing['recall10'] == 0
    assert missing['pool_conditional_ndcg10'] is None

def test_shared_prefix_truncation_and_paired_interval():
    operation = scifact_operation('claim', {'_id': 'doc', 'title': 'abcde', 'text': '123456'}, 3, 4)
    assert operation['state']['document']['title'] == 'abc'
    assert operation['state']['document']['text'] == '1234'
    result = bootstrap_pair([1, 0, 1], [1, 0, 1], 1000, 7)
    assert result['difference_ci95'] == [0, 0]
