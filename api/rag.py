"""
Vercel API RAG wrapper.

Goal:
- Reuse the existing RAG pipeline from src/rag_chatbot.py
- Load heavy models once (per serverless container lifecycle)
- Provide a simple function: chat(message) -> answer
"""

from __future__ import annotations

from typing import Any

from src.rag_chatbot import answer_question, load_rag_pipeline


_PIPELINE: tuple[Any, ...] | None = None


def get_pipeline() -> tuple[Any, ...]:
    """Load and cache the RAG pipeline objects."""
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

