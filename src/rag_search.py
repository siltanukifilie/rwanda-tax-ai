"""
Very simple RAG-style *retrieval* over your processed text files.

What it does:
- Loads all .txt files from data/processed
- Builds a lightweight keyword index with LlamaIndex (no embeddings / no API keys)
- Lets you type a question and prints a retrieved answer snippet

This is a beginner-friendly stepping stone toward a full RAG pipeline.
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices.keyword_table import KeywordTableIndex


INPUT_DIR = Path("data/processed")


def build_index() -> KeywordTableIndex:
    """Load documents from disk and build a keyword table index."""
    reader = SimpleDirectoryReader(input_dir=str(INPUT_DIR), required_exts=[".txt"])
    documents = reader.load_data()
    return KeywordTableIndex.from_documents(documents)


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Input folder not found: {INPUT_DIR.resolve()}")
        return

    print("Building index from .txt files in data/processed ...")
    index = build_index()
    query_engine = index.as_query_engine()
    print("Ready. Type a question (or press Enter to exit).")

    while True:
        question = input("\nQuestion> ").strip()
        if not question:
            break

        response = query_engine.query(question)
        print("\nAnswer:\n")
        print(str(response))


if __name__ == "__main__":
    main()

