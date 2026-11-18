import hashlib
import json
from pathlib import Path

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def order_key(seed, identifier):
    return digest([seed, str(identifier)])

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)

def bfcl_operation(row):
    tools = [{'tool_id': f'T{i}', 'name': f['name'], 'description': f.get('description', ''), 'parameters': f.get('parameters', {})} for i, f in enumerate(row['function'])]
    criteria = {t['tool_id']: 'Select the candidate tool named ' + t['name'] for t in tools}
    criteria['NO_TOOL'] = 'None of the provided tools is appropriate for the user request.'
    instructions = 'Select exactly one appropriate tool for this request, or NO_TOOL if none is appropriate. Evaluate tool purpose and relevant argument schema. Do not generate arguments or plan multiple calls.'
    return {'state': {'messages': row['question'][0], 'candidate_tools': tools}, 'questions': {'choice': {'type': 'choice', 'instructions': instructions, 'criteria': criteria}}}

RELEVANCE_RUBRIC = {
    '0': 'Not relevant: does not help assess the scientific claim.',
    '1': 'Weakly relevant: same broad topic, little evidence about the specific claim.',
    '2': 'Relevant: addresses part of the claim or provides indirect evidence.',
    '3': 'Highly relevant: directly provides scientific evidence assessing the claim, whether supporting or refuting it.',
}

def scifact_operation(claim, document, title_chars, text_chars):
    return {'state': {'scientific_claim': claim, 'document': {'id': document['_id'], 'title': document.get('title', '')[:title_chars], 'text': document.get('text', '')[:text_chars]}}, 'questions': {'choice': {'type': 'choice', 'instructions': 'Rate evidence relevance to the scientific claim on the fixed 0–3 scale. Relevance includes evidence that supports or refutes the claim; do not rate whether the claim itself is true.', 'criteria': RELEVANCE_RUBRIC}}}

def qwen_prompt(operation):
    return 'Evaluate this bounded operation. Use only the provided state and rubric. Return a JSON object with the single field choice.\n' + canonical(operation)
