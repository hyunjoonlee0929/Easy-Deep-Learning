"""Lightweight RAG pipeline with simple auto-evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RagResult:
    query: str
    answer: str
    contexts: list[str]
    scores: list[float]
    eval: dict[str, float]


def _split_docs(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    text = text.replace("\r", "\n")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks


def build_corpus(docs: list[str], chunk_size: int = 400, overlap: int = 80) -> list[str]:
    chunks: list[str] = []
    for doc in docs:
        chunks.extend(_split_docs(doc, chunk_size=chunk_size, overlap=overlap))
    return chunks


def retrieve(query: str, corpus: list[str], top_k: int = 3) -> tuple[list[str], list[float]]:
    if not corpus:
        return [], []
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(corpus + [query])
    doc_vecs = vectors[:-1]
    query_vec = vectors[-1]
    sims = cosine_similarity(query_vec, doc_vecs).ravel()
    top_idx = np.argsort(sims)[::-1][:top_k]
    contexts = [corpus[i] for i in top_idx]
    scores = [float(sims[i]) for i in top_idx]
    return contexts, scores


def _naive_answer(query: str, contexts: list[str]) -> str:
    if not contexts:
        return "No relevant context found."
    # Simple heuristic: return first sentence-like chunk from best context.
    best = contexts[0]
    return best[:400].strip()


def _auto_eval(query: str, answer: str, contexts: list[str]) -> dict[str, float]:
    if not contexts:
        return {"context_precision": 0.0, "context_recall": 0.0, "answer_overlap": 0.0}

    def _tokenize(s: str) -> set[str]:
        return set([t for t in s.lower().split() if t.isalnum()])

    q_tokens = _tokenize(query)
    a_tokens = _tokenize(answer)
    c_tokens = set()
    for c in contexts:
        c_tokens.update(_tokenize(c))

    precision = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
    recall = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
    overlap = len(a_tokens & c_tokens) / max(len(a_tokens), 1)
    return {
        "context_precision": float(precision),
        "context_recall": float(recall),
        "answer_overlap": float(overlap),
    }


def run_rag(
    query: str,
    docs: list[str],
    top_k: int = 3,
    chunk_size: int = 400,
    overlap: int = 80,
) -> RagResult:
    corpus = build_corpus(docs, chunk_size=chunk_size, overlap=overlap)
    contexts, scores = retrieve(query, corpus, top_k=top_k)
    answer = _naive_answer(query, contexts)
    eval_metrics = _auto_eval(query, answer, contexts)
    return RagResult(query=query, answer=answer, contexts=contexts, scores=scores, eval=eval_metrics)
