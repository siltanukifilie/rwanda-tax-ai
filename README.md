## Rwanda Tax AI — RAG App Scaffold

This repo is a starting folder structure for a Retrieval-Augmented Generation (RAG) app.

## Structure

- `data/raw/`: source documents (PDFs, docs, text files)
- `data/processed/`: derived artifacts (chunks, embeddings, vector index, metadata)
- `src/`: library code (loaders, chunking, indexing, retrieval, evaluation, etc.)
- `app.py`: entrypoint (currently a minimal scaffold)
- `requirements.txt`: Python dependencies

## Quickstart (Windows / PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Next steps

- Put documents into `data/raw/`
- Add modules under `src/` for:
  - loading + cleaning text
  - chunking
  - embedding
  - vector indexing + persistence into `data/processed/`
  - retrieval + answer generation

