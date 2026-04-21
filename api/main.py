from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.rag import chat, get_pipeline


# Vercel will look for a FastAPI app object in api/*.py
app = FastAPI(title="Rwanda Tax AI Assistant API", version="1.0.0")


# Allow browser-based frontend calls.
# For production, restrict this to your deployed frontend domain.
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


@app.get("/")
def root() -> dict:
    return {"message": "API is running"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.on_event("startup")
def _warmup() -> None:
    """Optional warmup to load models early (best-effort)."""
    try:
        get_pipeline()
    except Exception:
        pass


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    answer = chat(payload.message)
    return ChatResponse(answer=answer)

