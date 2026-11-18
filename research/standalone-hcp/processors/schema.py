from typing import Literal
from pydantic import BaseModel

class Decision(BaseModel):
    action: Literal['REPAIR', 'REGENERATE', 'DIAGNOSE_AND_REPAIR']
    explanation: str = ''
    latency_seconds: float = 0
    model_calls: int = 0
    raw: dict | str = ''

OPTIONS = {'REPAIR': 'Keep the current approach and fix the failing code.', 'REGENERATE': 'Discard the current code and implement a fresh solution from the requirements.'}
QUESTIONS = {'action': {'type': 'choice', 'instructions': 'Choose the strategy most likely to pass the tests on the next attempt.', 'criteria': OPTIONS}}
