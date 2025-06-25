from __future__ import annotations

from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None


def _ensure_model():
    global _tokenizer, _model  # pylint: disable=global-statement
    if _model is not None:
        return

    logger.info("Loading cross-encoder %s in fp16 …", _MODEL_NAME)
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, cache_dir="models/ms-marco-MiniLM-L-6-v2")
    _model = AutoModelForSequenceClassification.from_pretrained(
        _MODEL_NAME,
        cache_dir="models/ms-marco-MiniLM-L-6-v2"
    )
    _model.eval()
    logger.info("Cross-encoder loaded (device=%s)", _model.device)


def rerank_chunks(query: str, chunks: List[Dict], top_k: int = 10) -> List[Dict]:
    """Return *top_k* chunks ranked by cross-encoder relevance to *query*.

    Falls back to the input order if model loading fails.
    """
    if not chunks:
        return []

    try:
        _ensure_model()
    except Exception as exc:  # noqa: BLE001
        logger.error("Reranker model load failed: %s – returning input chunks", exc)
        return chunks[:top_k]

    assert _tokenizer is not None and _model is not None  # mypy hint

    pairs = [(
        query,
        f"{c.get('summary','')}\n{c.get('content','')}"[:512],  # truncate long text
    ) for c in chunks]

    inputs = _tokenizer.batch_encode_plus(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=256,
    )

    # Move tensors to same device as model
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        scores = _model(**inputs).logits.squeeze(-1)

    sorted_idx = torch.argsort(scores, descending=True).tolist()
    return [chunks[i] for i in sorted_idx[:top_k]] 