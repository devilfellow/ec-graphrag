from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+", re.UNICODE)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "with", "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня",
}
_SEMANTIC_MODEL: Any = None
_SEMANTIC_MODEL_NAME: str | None = None
_SEMANTIC_MODEL_UNAVAILABLE = False


def stable_id(prefix: str, *values: str) -> str:
    """Create a deterministic short identifier from stable string values."""
    payload = "\x1f".join(values).encode("utf-8")
    return f"{prefix}_{hashlib.sha1(payload).hexdigest()[:16]}"


def tokens(text: str) -> list[str]:
    """Tokenize text into lowercase word-like tokens."""
    return [token.lower() for token in TOKEN_RE.findall(text)]


def content_tokens(text: str) -> list[str]:
    """Tokenize text and remove short stopword tokens."""
    return [token for token in tokens(text) if token not in STOPWORDS and len(token) > 1]


def sentences(text: str) -> list[str]:
    """Split text into sentence-like spans."""
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


def estimate_tokens(text: str) -> int:
    """Estimate LLM token count from local tokenization."""
    return max(1, math.ceil(len(tokens(text)) * 1.25))


def lexical_similarity(left: str, right: str) -> float:
    """Compute cosine similarity over content-token counts."""
    a, b = Counter(content_tokens(left)), Counter(content_tokens(right))
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b)


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Compute normalized BM25 scores for a query over documents."""
    query_terms = content_tokens(query)
    tokenized = [content_tokens(document) for document in documents]
    if not query_terms or not tokenized:
        return [0.0] * len(documents)
    avg_len = sum(len(tokens_) for tokens_ in tokenized) / max(1, len(tokenized))
    document_frequency = Counter(
        term for tokens_ in tokenized for term in set(tokens_)
    )
    scores: list[float] = []
    for tokens_ in tokenized:
        frequencies = Counter(tokens_)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1 + (len(tokenized) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = frequency + k1 * (1 - b + b * len(tokens_) / max(1.0, avg_len))
            score += idf * frequency * (k1 + 1) / denominator
        scores.append(score)
    maximum = max(scores, default=0.0)
    return [score / maximum if maximum else 0.0 for score in scores]


def hashed_embedding(text: str, dimensions: int = 256) -> list[float]:
    """Create a deterministic hashed embedding used as semantic fallback."""
    vector = [0.0] * dimensions
    for token in content_tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % dimensions
        sign = 1.0 if digest[0] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine(left: list[float], right: list[float]) -> float:
    """Return non-negative cosine similarity for normalized vectors."""
    return max(0.0, sum(a * b for a, b in zip(left, right)))


def semantic_similarity_scores(
    query: str,
    documents: list[str],
    model_name: str | None = None,
    strict: bool = False,
) -> list[float]:
    """Score semantic similarity with Sentence Transformers or hashed fallback."""
    global _SEMANTIC_MODEL, _SEMANTIC_MODEL_NAME, _SEMANTIC_MODEL_UNAVAILABLE
    if not documents:
        return []
    if strict and _SEMANTIC_MODEL_UNAVAILABLE:
        raise RuntimeError(f"Embedding model is unavailable: {model_name}")
    if not _SEMANTIC_MODEL_UNAVAILABLE:
        try:
            requested_model = model_name or os.getenv(
                "ECGRAPHRAG_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
            if _SEMANTIC_MODEL is None or _SEMANTIC_MODEL_NAME != requested_model:
                from sentence_transformers import SentenceTransformer

                _SEMANTIC_MODEL = SentenceTransformer(requested_model)
                _SEMANTIC_MODEL_NAME = requested_model
            vectors = _SEMANTIC_MODEL.encode(
                [query, *documents],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_vector = vectors[0]
            return [
                max(0.0, float(sum(a * b for a, b in zip(query_vector, vector))))
                for vector in vectors[1:]
            ]
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            if strict:
                raise RuntimeError(f"Embedding model is unavailable: {model_name}") from exc
            _SEMANTIC_MODEL_UNAVAILABLE = True
    query_vector = hashed_embedding(query)
    return [cosine(query_vector, hashed_embedding(document)) for document in documents]


def semantic_backend() -> str:
    """Return the active semantic backend name for diagnostics."""
    if _SEMANTIC_MODEL is not None and _SEMANTIC_MODEL_NAME:
        return _SEMANTIC_MODEL_NAME
    return "hashed-fallback" if _SEMANTIC_MODEL_UNAVAILABLE else "not-loaded"
