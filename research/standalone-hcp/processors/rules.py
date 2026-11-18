from .schema import Decision

def decide(feedback):
    # Fixed before observing downstream outcomes; no learned or gold decisions.
    action = 'REGENERATE' if 'TIMEOUT' in feedback else 'REPAIR'
    return Decision(action=action, explanation='')
