"""
Load all .txt files from data/processed and split them into word-based chunks.

Requirements implemented:
- Uses LlamaIndex to load text files
- Chunks are ~300–500 words with ~50 words overlap
- Stores chunks in a list
- Prints number of chunks and one sample chunk
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core import SimpleDirectoryReader


INPUT_DIR = Path("data/processed")

# Chunking settings (word-based)
TARGET_WORDS_PER_CHUNK = 400  # aim for the middle of 300–500
MAX_WORDS_PER_CHUNK = 500
OVERLAP_WORDS = 50


def chunk_words(text: str) -> list[str]:
    """
    Split text into chunks using a sliding window over words.

    We aim for ~TARGET_WORDS_PER_CHUNK words, cap at MAX_WORDS_PER_CHUNK,
    and keep OVERLAP_WORDS between consecutive chunks to preserve context.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + MAX_WORDS_PER_CHUNK, len(words))

        # Prefer the target size when possible.
        preferred_end = min(start + TARGET_WORDS_PER_CHUNK, len(words))
        if preferred_end > start:
            end = max(preferred_end, end if end == len(words) else preferred_end)

        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        # Move the window forward, keeping overlap.
        start = max(end - OVERLAP_WORDS, start + 1)

    return chunks


def main() -> None:
    # Load all .txt files from the input folder.
    # SimpleDirectoryReader returns LlamaIndex Document objects (with .text).
    reader = SimpleDirectoryReader(input_dir=str(INPUT_DIR), required_exts=[".txt"])
    documents = reader.load_data()

    all_chunks: list[str] = []

    # Convert each document into chunks.
    for doc in documents:
        doc_text = (doc.text or "").strip()
        if not doc_text:
            continue
        all_chunks.extend(chunk_words(doc_text))

    # Print summary and one example.
    print(f"Chunks created: {len(all_chunks)}")

    if all_chunks:
        print("\nSample chunk:\n")
        print(all_chunks[0])


if __name__ == "__main__":
    main()

