import csv
import json
import re
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from experiments.affinity.common import scifact_operation, digest, file_hash, order_key

def tokenize(text):
    return re.findall(r'\w+', text.casefold(), flags=re.UNICODE)

def load(raw):
    raw = Path(raw)
    corpus = {str(r['_id']): r for r in map(json.loads, (raw / 'corpus.jsonl').read_text().splitlines())}
    queries = {str(r['_id']): r['text'] for r in map(json.loads, (raw / 'queries.jsonl').read_text().splitlines())}
    qrels = {}
    with (raw / 'qrels/test.tsv').open() as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            qid, did, score = str(row['query-id']), str(row['corpus-id']), int(row['score'])
            if qid not in queries or did not in corpus:
                raise ValueError('qrels ID absent from corpus/queries')
            if did in qrels.setdefault(qid, {}):
                raise ValueError('duplicate qrel pair')
            qrels[qid][did] = score
    return corpus, queries, qrels

def prepare(raw, config, fits):
    corpus, queries, qrels = load(raw)
    doc_ids = sorted(corpus)
    bm25 = BM25Okapi([tokenize(corpus[d].get('title', '') + ' ' + corpus[d]['text']) for d in doc_ids], k1=1.5, b=0.75, epsilon=0.25)
    query_ids = sorted(qrels, key=lambda q: order_key(config['seed'], q))[:config['scifact_queries']]
    pairs, pools = [], []
    for qid in query_ids:
        scores = bm25.get_scores(tokenize(queries[qid]))
        # Stable tie break by the sorted original string document ID, not qrels.
        ranked = np.argsort(-scores, kind='stable')[:config['scifact_candidates']]
        candidates = []
        for rank, index in enumerate(ranked, 1):
            did = doc_ids[index]
            operation = scifact_operation(queries[qid], corpus[did], config['document_title_chars'], config['document_text_chars'])
            token_info = fits(operation)  # No sample is dropped for retrieval coverage or model quality.
            pairs.append({'task_id': f'{qid}::{did}', 'query_id': qid, 'document_id': did, 'retrieval_rank': rank, 'operation': operation, 'input_state_hash': digest(operation), 'token_preflight': token_info})
            candidates.append({'document_id': did, 'retrieval_rank': rank, 'bm25_score': float(scores[index])})
        relevant = {d for d, grade in qrels[qid].items() if grade > 0}
        retrieved = {r['document_id'] for r in candidates}
        pools.append({'query_id': qid, 'claim': queries[qid], 'candidates': candidates, 'qrels': qrels[qid], 'relevant_in_pool': len(relevant & retrieved), 'relevant_total': len(relevant), 'candidate_recall': len(relevant & retrieved) / len(relevant) if relevant else None})
    files = [Path(raw) / n for n in ['corpus.jsonl', 'queries.jsonl', 'qrels/test.tsv']]
    return {'dataset': 'BEIR SciFact official test qrels', 'source': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip', 'archive_sha256': file_hash(Path(raw).parent / 'scifact.zip'), 'file_sha256': {str(p): file_hash(p) for p in files}, 'seed': config['seed'], 'selection': 'first 100 test query IDs by SHA256(seed,id), regardless of candidate recall', 'retrieval': {'model': 'rank-bm25 BM25Okapi CPU', 'k1': 1.5, 'b': 0.75, 'epsilon': 0.25, 'tokenization': 'Unicode regex word tokens, casefold; no stemming/stopwords', 'corpus': 'full title + text', 'top_k': config['scifact_candidates'], 'ties': 'original string document ID ascending'}, 'truncation': {'title_unicode_chars': config['document_title_chars'], 'text_unicode_chars': config['document_text_chars'], 'policy': 'prefix, identical text for both models; retrieval uses untruncated corpus'}, 'corpus_count': len(corpus), 'test_query_count': len(qrels), 'queries': pools, 'examples': pairs}
