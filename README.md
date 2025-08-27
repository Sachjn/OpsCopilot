# OpsCopilot — Minimal Working Project

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Render](https://img.shields.io/badge/deploy-render-brightgreen)](https://render.com) [![Vercel](https://img.shields.io/badge/deploy-vercel-blue)](https://vercel.com)

## Features
- PDF/Text ingestion with layout-agnostic chunking
- Semantic search via ChromaDB and SentenceTransformers
- LLM-backed contextual answers (optional OpenAI integration)
- Schema-driven JSON extraction API
- Simple React-like static frontend and Gradio demo for quick sharing



## Quick Start
```bash
# 1) Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# API: http://localhost:8000/docs

# (Optional) Enable LLM answers
export OPENAI_API_KEY=YOUR_KEY

# 2) Frontend
cd ../frontend
# Just open index.html with Live Server or any static server
```

## Docker

```bash
docker compose up --build
# Frontend: http://localhost:8080
# Backend:  http://localhost:8000/docs
```

## Use

1. Paste some text and click **Ingest**.
2. Ask a question in **Ask a Question**.
3. If `OPENAI_API_KEY` is set, you get grounded answers with citations; otherwise top chunks.

## Notes

* Vector store is persisted in `backend/chroma_data`.
* Swap embedding model via `EMBED_MODEL` env.
* Extend `/extract` with your JSON schemas for contracts, SOPs, etc.


## Pushing to GitHub (after unzipping)

```bash
cd ops-copilot
git init
git add .
git commit -m "Initial commit - OpsCopilot minimal"
git branch -M main
# Create a new repo on GitHub, then add remote:
git remote add origin https://github.com/<your-username>/OpsCopilot.git
git push -u origin main
```

## Quick Deploy Tips
- Render: Create a new Web Service, connect GitHub, set the root to `/backend`, use Dockerfile build or use the Python build with start command:
  `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Vercel: Deploy the `frontend` folder as a static site.
- If you don't want to use OpenAI keys in public repos, set them as environment variables in your host.

## Hugging Face Space (Optional Quick Demo)
You can run a Gradio demo that connects to the backend API (useful for quick public demos).

```
# From project root
cd hf_space
pip install -r requirements.txt
python app.py
# Visit http://localhost:7860
```

## One-Click Deploy Notes
- Render: after creating repo, link it and use `render.yaml` (edit repo URL).
- Vercel: import repo, choose `frontend` as the project root for static deploy.
- Hugging Face: create a new Space, choose Gradio, and upload contents of `hf_space` (set `BACKEND_API` environment variable if your backend is remote).
