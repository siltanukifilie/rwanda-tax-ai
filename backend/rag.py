"""
Backend RAG wrapper.

Goal:
- Reuse the existing RAG pipeline from src/rag_chatbot.py
- Load heavy models once (on server startup / first import)
- Provide a simple function: chat(message) -> response
"""

from __future__ import annotations

from typing import Any

from src.rag_chatbot import answer_question, load_rag_pipeline


# Load once and reuse (FAISS index + embedder + HF model are expensive to load)
_PIPELINE: tuple[Any, ...] | None = None


def get_pipeline() -> tuple[Any, ...]:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = load_rag_pipeline()
    return _PIPELINE


def chat(message: str) -> str:
    """Answer a user's message using the existing RAG system."""
    index, chunks, embedder, tokenizer, llm = get_pipeline()
    return answer_question(
        message,
        index=index,
        chunks=chunks,
        embedder=embedder,
        tokenizer=tokenizer,
        llm=llm,
    )

