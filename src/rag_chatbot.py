"""
Simple RAG chatbot using FAISS + sentence-transformers + a Hugging Face LLM.

Requirements implemented:
- Load FAISS index and chunks.pkl
- Use sentence-transformers "all-MiniLM-L6-v2" for query embedding
- Retrieve top 3 similar chunks from FAISS
- Use Hugging Face model "google/flan-t5-base"
- Build a prompt like:

  "Use the following context to answer the question:
   Context: {retrieved_chunks}
   Question: {user_question}"

- Generate response using the LLM
- Print final answer in a clean format

First run note:
- The LLM will be downloaded from Hugging Face the first time you run it.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Files produced by: python .\src\build_faiss_index.py
FAISS_INDEX_PATH = Path("faiss.index")
CHUNKS_PATH = Path("chunks.pkl")

# Retrieval settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

# LLM settings
LLM_MODEL_NAME = "google/flan-t5-base"
MAX_LENGTH = 300
MIN_LENGTH = 80
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.2

# Hard limits to prevent long-context crashes on small models/machines
TOP_CHUNKS_FOR_LLM = 2
MAX_CONTEXT_CHARS = 800


def load_rag_pipeline() -> tuple[faiss.Index, list[str], SentenceTransformer, AutoTokenizer, AutoModelForSeq2SeqLM]:
    """
    Load everything needed for RAG (index, chunks, embedder, tokenizer, LLM).

    The Streamlit UI should call this once and reuse the returned objects.
    """
    if not FAISS_INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Missing FAISS artifacts. Build them first with: python .\\src\\build_faiss_index.py"
        )

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    chunks = load_chunks(CHUNKS_PATH)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    llm = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

    return index, chunks, embedder, tokenizer, llm


def answer_question(
    user_question: str,
    *,
    index: faiss.Index,
    chunks: list[str],
    embedder: SentenceTransformer,
    tokenizer: AutoTokenizer,
    llm: AutoModelForSeq2SeqLM,
) -> str:
    """Run retrieval + prompt + generation and return the final answer text."""
    query_vec = embed_query(embedder, user_question)
    retrieved_chunks = retrieve_top_chunks(index, chunks, query_vec, top_k=TOP_K)
    prompt = build_prompt(retrieved_chunks=retrieved_chunks, user_question=user_question)
    return generate_response(tokenizer=tokenizer, model=llm, prompt=prompt)


def load_chunks(path: Path) -> list[str]:
    """Load a pickled list of chunk texts (list[str])."""
    with path.open("rb") as f:
        chunks = pickle.load(f)

    if not isinstance(chunks, list) or not all(isinstance(x, str) for x in chunks):
        raise TypeError("chunks.pkl must contain a Python list of strings.")

    return chunks


def embed_query(model: SentenceTransformer, text: str) -> np.ndarray:
    """
    Convert the user question into an embedding vector for FAISS search.

    Returns a float32 numpy array of shape: (1, embedding_dim)
    """
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype("float32")


def retrieve_top_chunks(index: faiss.Index, chunks: list[str], query_vec: np.ndarray, top_k: int = TOP_K) -> list[str]:
    """Search FAISS and return the top-k chunk texts."""
    _distances, indices = index.search(query_vec, top_k)

    results: list[str] = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
    return results


def _clean_context_text(text: str) -> str:
    """
    Basic cleanup to reduce duplicated lines and common boilerplate.

    This keeps the logic simple and beginner-friendly:
    - trims whitespace
    - removes repeated identical lines (often headers/footers)
    - drops very short/empty lines
    """
    lines = [ln.strip() for ln in text.splitlines()]

    cleaned: list[str] = []
    seen: set[str] = set()

    for ln in lines:
        if not ln:
            continue

        # Skip tiny lines that are often page numbers or artifacts.
        if len(ln) <= 2 and ln.isdigit():
            continue

        key = ln.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(ln)

    return "\n".join(cleaned).strip()


def build_prompt(retrieved_chunks: list[str], user_question: str) -> str:
    """Build the prompt in the exact format requested."""
    # ONLY keep top 2–3 chunks (we use 2 by default for safety)
    top_chunks = retrieved_chunks[:TOP_CHUNKS_FOR_LLM]

    # JOIN + TRIM HARD (VERY IMPORTANT FIX)
    context = "\n\n".join(top_chunks)
    context = _clean_context_text(context)
    context = context[:MAX_CONTEXT_CHARS]

    return f"""
You are a Rwanda Tax Expert AI assistant.

You help users understand Rwanda tax topics using retrieved knowledge (RAG) and general knowledge.

RULES:
1. The retrieved context is the ONLY source of factual correctness.
2. Extract key facts from the context. Do NOT copy it verbatim.
3. Use your general knowledge ONLY to explain, simplify, and give examples.
4. Never contradict the retrieved context.
5. If context is missing information, clearly explain what is known and what is general knowledge.
6. Remove repetition, boilerplate, and duplicated text.

OUTPUT FORMAT:
- Title
- Clear Explanation
- Bullet Points (if useful)
- Simple Real-World Example (if applicable)

STYLE:
- Simple and human
- No raw retrieval text
- No repetition
- Structured like ChatGPT answers

Context:
{context}

Question:
{user_question}

Answer:
"""


def generate_response(tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, prompt: str) -> str:
    """
    Generate a response from the LLM.

    We keep the generation settings simple for beginners.
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    output_ids = model.generate(
        **inputs,
        # Generation settings to reduce repetition and improve readability
        max_length=MAX_LENGTH,
        min_length=MIN_LENGTH,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.pad_token_id,
    )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Many models include the prompt in the output. Remove it if present.
    if full_text.startswith(prompt):
        full_text = full_text[len(prompt) :]

    return full_text.strip()


def main() -> None:
    # 1) Check required files exist
    if not FAISS_INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        print("Missing FAISS artifacts.")
        print(f"- faiss.index exists? {FAISS_INDEX_PATH.exists()}")
        print(f"- chunks.pkl exists?  {CHUNKS_PATH.exists()}")
        print("\nBuild them first by running:")
        print(r"  python .\src\build_faiss_index.py")
        return

    # 2) Load everything needed for RAG
    print("Loading RAG pipeline ...")
    index, chunks, embedder, tokenizer, llm = load_rag_pipeline()

    print("\nRAG chatbot is ready.")
    print("Type your question. Type 'exit' to quit.\n")

    while True:
        user_question = input("You> ").strip()
        if not user_question:
            continue
        if user_question.lower() in {"exit", "quit"}:
            break

        # 5) Answer using the RAG pipeline
        final_answer = answer_question(
            user_question,
            index=index,
            chunks=chunks,
            embedder=embedder,
            tokenizer=tokenizer,
            llm=llm,
        )

        # 7) Print the final answer in a clean format
        print("\nAnswer:\n")
        print(final_answer if final_answer else "(No answer generated.)")
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()

