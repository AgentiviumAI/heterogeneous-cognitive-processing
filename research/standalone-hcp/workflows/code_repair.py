from processors import rules

def state_for(task, code, evaluation):
    return {'requirement': task['prompt'], 'code': code, 'test_feedback': evaluation['feedback'][-2000:]}

def decision(variant, qwen, d1, state, seed):
    if variant == 'A':
        return qwen.decide(state, seed, bounded=False)
    if variant == 'B':
        return d1.decide(state)
    if variant == 'C':
        return rules.decide(state['test_feedback'])
    if variant == 'D':
        return qwen.decide(state, seed, bounded=True)
    raise ValueError(variant)
