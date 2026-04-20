"""
Build a FAISS index + chunks file from your processed .txt documents.

Inputs:
- data/processed/*.txt   (text extracted from PDFs)

Outputs (saved in the project root):
- chunks.pkl   (Python list[str] of chunk texts)
- faiss.index  (FAISS index built from chunk embeddings)

This script is intentionally beginner-friendly and well commented.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from llama_index.core import SimpleDirectoryReader
from sentence_transformers import SentenceTransformer


# Where your text files live
INPUT_DIR = Path("data/processed")

# Output files (written in the project root)
CHUNKS_PATH = Path("chunks.pkl")
FAISS_INDEX_PATH = Path("faiss.index")

# Embedding model (must match what you use at query time)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking settings (word-based)
WORDS_PER_CHUNK = 400  # in the 300–500 range
OVERLAP_WORDS = 50


def chunk_words(text: str, words_per_chunk: int = WORDS_PER_CHUNK, overlap_words: int = OVERLAP_WORDS) -> list[str]:
    """
    Split text into overlapping word chunks.

    Example with words_per_chunk=400 and overlap_words=50:
    - chunk1: words 1..400
    - chunk2: words 351..750
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        # Slide the window forward, keeping overlap_words words.
        start = max(end - overlap_words, start + 1)

    return chunks


def load_all_texts() -> list[str]:
    """Load all .txt files from INPUT_DIR and return their full text content."""
    reader = SimpleDirectoryReader(input_dir=str(INPUT_DIR), required_exts=[".txt"])
    docs = reader.load_data()
    return [(d.text or "").strip() for d in docs if (d.text or "").strip()]


def main() -> None:
    # 1) Make sure the input folder exists
    if not INPUT_DIR.exists():
        print(f"Input folder not found: {INPUT_DIR.resolve()}")
        return

    # 2) Load all .txt files
    print(f"Loading .txt files from: {INPUT_DIR.resolve()}")
    texts = load_all_texts()
    if not texts:
        print("No .txt files found (or they were empty). Nothing to index.")
        return

    # 3) Split all texts into chunks
    print("Chunking texts ...")
    chunks: list[str] = []
    for text in texts:
        chunks.extend(chunk_words(text))

    print(f"Total chunks: {len(chunks)}")
    if not chunks:
        print("No chunks created. Nothing to index.")
        return

    # 4) Embed chunks
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding chunks (this can take a bit) ...")
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,  # good for cosine similarity
        show_progress_bar=True,
    ).astype("float32")

    # 5) Build the FAISS index
    # For normalized vectors, inner product (dot product) corresponds to cosine similarity.
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # 6) Save artifacts
    with CHUNKS_PATH.open("wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print(f"Saved chunks to: {CHUNKS_PATH.resolve()}")
    print(f"Saved FAISS index to: {FAISS_INDEX_PATH.resolve()}")
    print("Index build complete")


if __name__ == "__main__":
    main()

