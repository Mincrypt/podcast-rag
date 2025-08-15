from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import numpy as np

def reciprocal_rank_fusion(scores_list, k=60):
    fused = {}
    for ranks in scores_list:
        for doc_id, rank in ranks.items():
            fused[doc_id] = fused.get(doc_id, 0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

def hybrid_rank(chroma_res, bm25_model: BM25Okapi, corpus_ids, query, top_k=8):
    chroma_ids = chroma_res["ids"][0]
    chroma_ranks = {doc_id: i+1 for i, doc_id in enumerate(chroma_ids)}

    bm25_scores = bm25_model.get_scores(query.lower().split())
    # rank: 1 is best (largest score)
    order = np.argsort(-bm25_scores)
    bm25_ranks = {corpus_ids[i]: int(np.where(order==i)[0][0]) + 1 for i in range(len(corpus_ids))}

    fused = reciprocal_rank_fusion([chroma_ranks, bm25_ranks])
    return [doc_id for doc_id, _ in fused[:top_k]]
