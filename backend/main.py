from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.rag import chat, get_pipeline


app = FastAPI(title="Rwanda Tax AI Assistant API", version="1.0.0")


# Allow frontend to call the API from a different origin during development.
# In production, replace "*" with your real domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User question/message")


class ChatResponse(BaseModel):
    answer: str


@app.on_event("startup")
def _warmup() -> None:
    """
    Optional warmup so the first chat request is fast.

    This loads:
    - FAISS index
    - sentence-transformers embedder
    - Hugging Face model/tokenizer
    """
    try:
        get_pipeline()
    except Exception:
        # Don't crash the server at startup; errors will surface on /chat.
        pass


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    answer = chat(payload.message)
    return ChatResponse(answer=answer)

