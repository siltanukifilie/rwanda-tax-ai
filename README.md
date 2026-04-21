## Rwanda Tax AI — RAG App Scaffold

This repo is a starting folder structure for a Retrieval-Augmented Generation (RAG) app.

## Structure

- `data/raw/`: source documents (PDFs, docs, text files)
- `data/processed/`: derived artifacts (chunks, embeddings, vector index, metadata)
- `src/`: library code (loaders, chunking, indexing, retrieval, evaluation, etc.)
- `app.py`: Streamlit UI (optional)
- `api/`: FastAPI backend (Vercel-compatible)
- `frontend/`: ChatGPT-style static frontend (HTML/CSS/JS)
- `requirements.txt`: Python dependencies

## Quickstart (Windows / PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Full-stack ChatGPT-style web app (FastAPI + HTML/CSS/JS)

### 1) Build the FAISS artifacts (only once, or whenever documents change)

Make sure you already extracted PDFs to `data/processed/*.txt`, then:

```bash
python .\src\build_faiss_index.py
```

This produces:
- `chunks.pkl`
- `faiss.index`

### 2) Start the backend (FastAPI)

```bash
python -m pip install -r requirements.txt
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Health check: `GET http://127.0.0.1:8000/health`

### 3) Open the frontend

Open `frontend/index.html` in your browser.

If your backend URL is different, edit `API_URL` in `frontend/app.js`.

API response format used by the frontend: `{ "answer": "..." }`

## Next steps

- Put documents into `data/raw/`
- Add modules under `src/` for:
  - loading + cleaning text
  - chunking
  - embedding
  - vector indexing + persistence into `data/processed/`
  - retrieval + answer generation

