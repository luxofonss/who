from __future__ import annotations

from typing import List, Dict

from rank_bm25 import BM25Okapi


def bm25_search(query: str, metadata: List[Dict], top_k: int = 10) -> List[Dict]:
    """Simple BM25 search over chunk texts (summary + content)."""
    texts = [f"{c.get('summary','')}\n{c.get('content','')}" for c in metadata]
    tokenised = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenised)
    scores = bm25.get_scores(query.split())
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [metadata[i] for i in ranked] 