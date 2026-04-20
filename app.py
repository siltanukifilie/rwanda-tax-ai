from __future__ import annotations

import os
from pathlib import Path


DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")


def main() -> None:
    """
    Minimal entrypoint for a future RAG app.

    Next steps:
    - Load documents from data/raw
    - Chunk + embed
    - Store in a vector index under data/processed
    - Retrieve + generate answers from a chosen LLM
    """
    print("Rwanda Tax AI (RAG) project scaffold is ready.")
    print(f"Raw data folder: {DATA_RAW_DIR.resolve()}")
    print(f"Processed data folder: {DATA_PROCESSED_DIR.resolve()}")
    print(f"Python: {os.sys.version.split()[0]}")


if __name__ == "__main__":
    main()

