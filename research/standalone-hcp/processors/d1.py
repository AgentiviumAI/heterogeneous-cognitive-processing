import time
import httpx
from .schema import Decision, QUESTIONS

class D1:
    def __init__(self, config):
        self.client = httpx.Client(base_url=config['d1_url'], timeout=180, trust_env=False)

    def decide(self, state):
        start = time.perf_counter()
        request_body = {'state': state, 'questions': QUESTIONS}
        response = self.client.post('/v1/systemone', json=request_body)
        response.raise_for_status()
        raw = response.json()
        return Decision(action=raw['answers']['action']['choice'], latency_seconds=time.perf_counter() - start, model_calls=1, raw={'request': request_body, 'response': raw})
