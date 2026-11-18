import json
import re
import time
import httpx
from .schema import Decision, OPTIONS

def extract_code(text):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
    return (max(blocks, key=len) if blocks else text).strip()

class Qwen:
    def __init__(self, config):
        self.config = config
        self.client = httpx.Client(base_url=config['qwen_url'], timeout=180, trust_env=False)

    def call(self, prompt, seed, max_tokens=None, schema=None):
        body = {'model': self.config['qwen_model'], 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0, 'seed': seed, 'max_tokens': max_tokens or self.config['max_output_tokens'], 'chat_template_kwargs': {'enable_thinking': False}}
        if schema:
            body['guided_json'] = schema
        start = time.perf_counter()
        r = self.client.post('/v1/chat/completions', json=body)
        r.raise_for_status()
        data = r.json()
        return {'content': data['choices'][0]['message']['content'], 'finish_reason': data['choices'][0]['finish_reason'], 'usage': data['usage'], 'latency_seconds': time.perf_counter() - start, 'model_calls': 1, 'seed': seed, 'request': body}

    def generate(self, task, seed, previous=None, feedback=None, decision=None):
        prompt = self.generation_prompt(task, previous, feedback, decision)
        result = self.call(prompt, seed)
        result['code'] = extract_code(result['content'])
        return result

    def generation_prompt(self, task, previous=None, feedback=None, decision=None):
        prompt = 'Write a complete Python solution. Output only Python code, including required imports. Do not execute shell commands.\nRequirement:\n' + task['prompt'] + '\nRequired public tests:\n' + '\n'.join(task['tests'])
        if previous is not None and decision.action != 'REGENERATE':
            prompt += '\nCurrent code:\n' + previous + '\nObserved test failure:\n' + feedback + '\nRepair guidance:\n' + (decision.explanation or 'Fix the failing solution while preserving its working parts.')
        elif previous is not None:
            prompt += '\nThe previous solution failed. Implement a fresh approach. Observed failure:\n' + feedback
        return prompt

    def generation_request(self, task, seed):
        return {'model': self.config['qwen_model'], 'messages': [{'role': 'user', 'content': self.generation_prompt(task)}], 'temperature': 0, 'seed': seed, 'max_tokens': self.config['max_output_tokens'], 'chat_template_kwargs': {'enable_thinking': False}}

    def decide(self, state, seed, bounded):
        if bounded:
            schema = {'type': 'object', 'properties': {'action': {'type': 'string', 'enum': list(OPTIONS)}}, 'required': ['action'], 'additionalProperties': False}
            result = self.call('Select the next repair strategy for this state. Return JSON.\n' + json.dumps({'state': state, 'questions': {'action': {'type': 'choice', 'instructions': 'Choose the strategy most likely to pass the tests on the next attempt.', 'criteria': OPTIONS}}}), seed, 64, schema)
            return Decision(action=json.loads(result['content'])['action'], latency_seconds=result['latency_seconds'], model_calls=1, raw=result)
        result = self.call('Diagnose the failed Python solution and explain how to fix it. Do not write code.\n' + json.dumps(state), seed, 256)
        return Decision(action='DIAGNOSE_AND_REPAIR', explanation=result['content'], latency_seconds=result['latency_seconds'], model_calls=1, raw=result)
