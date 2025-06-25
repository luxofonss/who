from __future__ import annotations

from typing import List, Literal
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from loguru import logger

# Global model/tokenizer cache
_TOKENIZER: AutoTokenizer | None = None
_MODEL: AutoModel | None = None

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


def _load_model(model_name: str = "Salesforce/codet5-base") -> tuple[AutoTokenizer, AutoModel]:
    global _TOKENIZER, _MODEL

    if _TOKENIZER is None or _MODEL is None:
        logger.info(f"Loading HuggingFace model and tokenizer ({model_name}) …")
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name, cache_dir="models/e5-base-code")
        _MODEL = AutoModel.from_pretrained(model_name, cache_dir="models/e5-base-code", device_map="auto")
        _MODEL.eval().to(DEVICE)  # Move model to GPU/CPU
    return _TOKENIZER, _MODEL


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def _cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]  # CLS token (first position)


def embed_texts(
    texts: List[str],
    model_name: str = "Salesforce/codet5-base",
    pooling: Literal["mean", "cls"] = "mean"
) -> np.ndarray:
    """Return a (n, 768) NumPy array of embeddings for a list of code/text strings."""
    if not texts:
        return np.empty((0, 768))

    tokenizer, model = _load_model(model_name)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    if pooling == "mean":
        embeddings = _mean_pooling(outputs, inputs["attention_mask"])
    elif pooling == "cls":
        embeddings = _cls_pooling(outputs)
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")

    return embeddings.cpu().numpy()


def embed(
    text: str,
    model_name: str = "Salesforce/codet5-base",
    pooling: Literal["mean", "cls"] = "mean"
) -> np.ndarray:
    """Return a 1D embedding vector for a single text or code snippet."""
    return embed_texts([text], model_name, pooling).reshape(-1)


def embed_chunk(chunk: dict) -> np.ndarray:
    """Embed summary + content of a chunk."""
    text = f"{chunk.get('summary','')}\n\n{chunk.get('content','')}"
    return embed_texts([text]).reshape(-1)
