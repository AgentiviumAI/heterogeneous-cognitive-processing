import math

def ranking_metrics(ranking, qrels, cutoff=10, pool=None):
    positive = {d: grade for d, grade in qrels.items() if grade > 0}
    gains = [(2 ** positive.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranking[:cutoff])]
    ideal = sorted(positive.values(), reverse=True)[:cutoff]
    idcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
    rr = next((1 / (i + 1) for i, d in enumerate(ranking[:cutoff]) if d in positive), 0)
    recall = len(set(ranking[:cutoff]) & positive.keys()) / len(positive) if positive else None
    result = {'ndcg10': sum(gains) / idcg if idcg else None, 'mrr10': rr, 'recall10': recall}
    if pool is not None:
        pool_ideal = sorted([positive[d] for d in pool if d in positive], reverse=True)[:cutoff]
        pool_idcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(pool_ideal))
        result['pool_conditional_ndcg10'] = sum(gains) / pool_idcg if pool_idcg else None
    return result
