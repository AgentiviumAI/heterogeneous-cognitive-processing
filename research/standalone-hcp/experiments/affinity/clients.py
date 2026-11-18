import json
import resource
import time
from pathlib import Path
import httpx
from pydantic import BaseModel, field_validator
from processors.qwen import Qwen
from processors.d1 import D1
from .common import canonical, qwen_prompt

class Choice(BaseModel):
    choice: str

    @field_validator('choice', mode='before')
    @classmethod
    def exact_string(cls, value):
        if not isinstance(value, str):
            raise ValueError('choice must be a string')
        return value

def cgroup_snapshot():
    result = {'cpu_usage_usec': None, 'memory_current_bytes': None}
    try:
        cpu = dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
        result['cpu_usage_usec'] = int(cpu['usage_usec'])
        result['memory_current_bytes'] = int(Path('/sys/fs/cgroup/memory.current').read_text())
    except (OSError, ValueError, KeyError):
        pass
    return result

class Clients:
    def __init__(self, config):
        self.config = config
        self.qwen = Qwen({**config, 'max_output_tokens': 32})
        self.d1 = D1(config)
        for client in [self.qwen.client, self.d1.client]:
            client.timeout = httpx.Timeout(60)

    def health(self):
        for client in [self.qwen.client, self.d1.client]:
            client.get('/health').raise_for_status()

    def fits(self, operation):
        state = canonical(operation)
        if len(state) > self.config['state_max_chars']:
            raise ValueError('shared operation exceeds predeclared character envelope')
        q = self.qwen.client.post('/tokenize', json={'model': self.config['qwen_model'], 'prompt': qwen_prompt(operation)})
        d = self.d1.client.post('/tokenize', json={'content': state})
        q.raise_for_status(); d.raise_for_status()
        counts = {'qwen_surrogate_tokens': q.json()['count'], 'd1_surrogate_tokens': len(d.json()['tokens']), 'reserve': self.config['token_reserve']}
        if counts['qwen_surrogate_tokens'] + counts['reserve'] >= self.config['qwen_token_limit'] or counts['d1_surrogate_tokens'] + counts['reserve'] >= self.config['d1_token_limit']:
            raise ValueError('shared operation exceeds predeclared tokenizer envelope')
        return counts

    def execute(self, processor, operation):
        before = cgroup_snapshot()
        cpu_before = resource.getrusage(resource.RUSAGE_SELF)
        prepare_start = time.perf_counter()
        choices = list(operation['questions']['choice']['criteria'])
        prompt = qwen_prompt(operation)
        schema = {'type': 'object', 'properties': {'choice': {'type': 'string', 'enum': choices}}, 'required': ['choice'], 'additionalProperties': False}
        # The same canonical state text and same bounded rubric are sent through each native protocol.
        body = {'state': canonical(operation['state']), 'questions': operation['questions']}
        preparation_ms = (time.perf_counter() - prepare_start) * 1000
        start_utc, started = time.time(), time.perf_counter()
        raw, choice, valid, error, usage, finish_reason = None, None, False, None, {}, None
        try:
            if processor == 'qwen3-8b':
                result = self.qwen.call(prompt, self.config['seed'], 32, schema)
                latency_ms = result['latency_seconds'] * 1000
                raw, usage, finish_reason = result, result['usage'], result['finish_reason']
                parsed = Choice.model_validate(json.loads(result['content']))
                choice = parsed.choice
                if finish_reason != 'stop':
                    raise ValueError('non-stop finish reason')
            else:
                response = self.d1.client.post('/v1/systemone', json=body)
                response.raise_for_status()
                raw = response.json()
                latency_ms = (time.perf_counter() - started) * 1000
                usage = raw.get('usage', {})
                choice = Choice.model_validate({'choice': raw['answers']['choice']['choice']}).choice
            if choice not in choices:
                raise ValueError('choice outside frozen contract')
            reported_input = usage.get('prompt_tokens', usage.get('input_tokens'))
            limit = self.config['qwen_token_limit'] if processor == 'qwen3-8b' else self.config['d1_token_limit']
            if reported_input is not None and reported_input >= limit:
                raise ValueError('input reaches context limit; possible silent truncation')
            valid = True
        except Exception as exception:
            latency_ms = (time.perf_counter() - started) * 1000
            error = {'type': type(exception).__name__, 'message': str(exception)}
            if isinstance(exception, httpx.HTTPStatusError):
                raw = exception.response.text
        ended = time.time()
        after = cgroup_snapshot()
        cpu_after = resource.getrusage(resource.RUSAGE_SELF)
        delta = after['cpu_usage_usec'] - before['cpu_usage_usec'] if before['cpu_usage_usec'] is not None and after['cpu_usage_usec'] is not None else None
        return {'decision': choice if valid else None, 'valid': valid, 'error': error, 'latency_ms': latency_ms, 'input_preparation_ms': preparation_ms, 'request_start_utc': start_utc, 'request_end_utc': ended, 'input_tokens': usage.get('prompt_tokens', usage.get('input_tokens')), 'output_tokens': usage.get('completion_tokens', usage.get('output_tokens')), 'raw_response': raw, 'finish_reason': finish_reason, 'model_request_count': 1, 'harness_cpu_seconds': cpu_after.ru_utime + cpu_after.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime, 'harness_max_rss_kib': cpu_after.ru_maxrss, 'container_cpu_usage_usec_delta': delta, 'container_memory_before_bytes': before['memory_current_bytes'], 'container_memory_after_bytes': after['memory_current_bytes'], 'gpu_time_seconds': None}
