"""
Simple RAG retrieval script using FAISS + sentence-transformers.

It expects two files (in the project root by default):
- faiss.index : the FAISS vector index
- chunks.pkl  : a pickled Python list of text chunks (strings)

What it does:
- Ask the user for a question
- Embed the question using: all-MiniLM-L6-v2
- Search FAISS for the top 3 most similar chunks
- Print the question + the 3 retrieved chunks
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Files to load (change these paths if your files live elsewhere)
FAISS_INDEX_PATH = Path("faiss.index")
CHUNKS_PATH = Path("chunks.pkl")

# Embedding model to use
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# How many chunks to retrieve
TOP_K = 3


def load_chunks(path: Path) -> list[str]:
    """Load a pickled list of text chunks (strings)."""
    with path.open("rb") as f:
        chunks = pickle.load(f)

    if not isinstance(chunks, list) or not all(isinstance(x, str) for x in chunks):
        raise TypeError("chunks.pkl must contain a Python list of strings.")

    return chunks


def embed_text(model: SentenceTransformer, text: str) -> np.ndarray:
    """
    Convert text into a single embedding vector.

    FAISS expects float32 numpy arrays.
    Shape returned: (1, embedding_dim)
    """
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype("float32")


def main() -> None:
    # 1) Check that required files exist
    if not FAISS_INDEX_PATH.exists():
        print(f"FAISS index not found: {FAISS_INDEX_PATH.resolve()}")
        print("Expected a file named 'faiss.index'.")
        return

    if not CHUNKS_PATH.exists():
        print(f"Chunks file not found: {CHUNKS_PATH.resolve()}")
        print("Expected a file named 'chunks.pkl'.")
        return

    # 2) Load the FAISS index and the chunks list
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    chunks = load_chunks(CHUNKS_PATH)

    # 3) Load the embedding model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 4) Ask the user for a question
    question = input("Enter your question: ").strip()
    if not question:
        print("No question provided. Exiting.")
        return

    # 5) Embed the question
    query_vector = embed_text(model, question)

    # 6) Search FAISS for the nearest neighbors
    # D: distances/similarity scores, I: indices into the 'chunks' list
    D, I = index.search(query_vector, TOP_K)

    # 7) Print results
    print("\nUser question:")
    print(question)

    print(f"\nTop {TOP_K} retrieved chunks:\n")
    for rank, chunk_idx in enumerate(I[0], start=1):
        # FAISS can return -1 if it fails to find neighbors (rare, but safe to handle)
        if chunk_idx < 0 or chunk_idx >= len(chunks):
            print(f"[{rank}] (no chunk found)")
            print("-" * 60)
            continue

        print(f"[{rank}] (chunk #{chunk_idx})")
        print(chunks[chunk_idx])
        print("-" * 60)


if __name__ == "__main__":
    main()

