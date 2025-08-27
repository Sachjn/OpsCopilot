import os
import io
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Embeddings & Vector Store
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Optional LLM (OpenAI) - safe client setup and fallback handling
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oai_client = None
if OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        oai_client = openai
    except Exception:
        # If modern OpenAI package isn't available or different, we'll attempt to import OpenAI client class
        try:
            from openai import OpenAI as OpenAIClient
            oai_client = OpenAIClient(api_key=OPENAI_API_KEY)
        except Exception:
            oai_client = None

from .utils import extract_text_from_pdf, chunk_text
from .store import get_collection

app = FastAPI(title="OpsCopilot — Minimal")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model once
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
emb_model = SentenceTransformer(EMBED_MODEL_NAME)

# --------- Models ---------
class IngestTextReq(BaseModel):
    doc_id: Optional[str] = None
    title: str
    text: str
    metadata: Dict[str, Any] = {}

class QueryReq(BaseModel):
    query: str
    top_k: int = 4

class QueryResp(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    mode: str

class ExtractReq(BaseModel):
    document_id: str
    schema: Dict[str, Any]

class ExtractResp(BaseModel):
    data: Dict[str, Any]
    confidence: float
    spans: List[Dict[str, Any]]

# --------- Helpers ---------

def call_llm_chat(messages: list, model: str = None, temperature: float = 0.0):
    """Call the OpenAI chat API in a resilient way. Returns text content or raises."""
    if oai_client is None:
        raise RuntimeError("OpenAI client not configured")
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Try modern openai.ChatCompletion
    try:
        resp = oai_client.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
        # Try to extract content from known shapes
        choice = resp.choices[0]
        if hasattr(choice, 'message'):
            return choice.message.get('content') if isinstance(choice.message, dict) else choice.message.content
        elif 'message' in choice:
            return choice['message'].get('content')
        elif 'text' in choice:
            return choice['text']
    except Exception:
        pass
    # Try the newer client style (openai.ChatCompletion or OpenAI().chat.completions.create)
    try:
        resp = oai_client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        # Different clients return different shapes
        if hasattr(resp, 'choices'):
            choice = resp.choices[0]
            if hasattr(choice, 'message'):
                return choice.message.content
            elif 'message' in choice:
                return choice['message'].get('content')
            elif 'text' in choice:
                return choice['text']
        elif isinstance(resp, dict) and 'choices' in resp:
            ch = resp['choices'][0]
            if isinstance(ch, dict) and 'message' in ch:
                return ch['message'].get('content')
            return ch.get('text')
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")


def embed_texts(texts: List[str]):
    return emb_model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

# --------- Routes ---------
@app.get("/")
def root():
    return {"ok": True, "service": "OpsCopilot Minimal", "llm": bool(OPENAI_API_KEY)}

@app.post("/ingest/text")
def ingest_text(req: IngestTextReq):
    col = get_collection()
    chunks = chunk_text(req.text)
    ids = [f"{req.doc_id or req.title}:{i}" for i in range(len(chunks))]
    metas = [{"title": req.title, **req.metadata, "chunk_index": i} for i in range(len(chunks))]
    embeddings = embed_texts(chunks)

    col.add(documents=chunks, metadatas=metas, ids=ids, embeddings=embeddings)

    return {"ingested_chunks": len(chunks), "document_id": req.doc_id or req.title}

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...), title: str = Form(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
    content = await file.read()
    text = extract_text_from_pdf(io.BytesIO(content))
    return ingest_text(IngestTextReq(title=title, text=text))

@app.post("/query", response_model=QueryResp)
def query(req: QueryReq):
    col = get_collection()
    # Retrieve
    results = col.query(query_texts=[req.query], n_results=req.top_k, include=["documents", "metadatas", "distances"])
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Build context string
    context = "\n\n".join([f"[Chunk {i}] {d}" for i, d in enumerate(docs)])

    # If OpenAI available, generate; else return extractive context
    if OPENAI_API_KEY:
        system = (
            "You are OpsCopilot. Answer ONLY from the provided context. "
            "Cite chunk indices like [Chunk i]. If unknown, say you don't know."
        )
        prompt = f"Context:\n{context}\n\nQuestion: {req.query}\nAnswer:"
        answer_text = call_llm_chat([{"role":"system","content": system},{"role":"user","content": prompt}], model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'), temperature=0.1)
        mode = 'llm'
    else:
        # No LLM: return top chunks concatenated with a tip
        answer_text = (
            "\n".join(docs[:2]) +
            "\n\n(Note: LLM not configured — returning best-matching context. Set OPENAI_API_KEY to enable generation.)"
        )
        mode = "context"

    citations = [
        {"title": m.get("title"), "chunk_index": m.get("chunk_index"), "preview": d[:200]}
        for m, d in zip(metas, docs)
    ]
    return QueryResp(answer=answer_text, citations=citations, mode=mode)

@app.post("/extract", response_model=ExtractResp)
def extract(req: ExtractReq):
    """Schema-driven extraction. Uses LLM when available; naive regex fallback otherwise."""
    col = get_collection()
    # Get all chunks for the document
    results = col.get(where={"title": req.document_id})
    docs = results.get("documents", [])

    if not docs:
        raise HTTPException(status_code=404, detail="Document not found in index. Use /ingest first.")

    joined = "\n\n".join(docs)

    if OPENAI_API_KEY:
        system = (
            "Extract a JSON object strictly matching the provided JSON Schema. "
            "Use only information present in the text. If a field is missing, use null."
        )
        user = (
            f"Text:\n{joined[:15000]}\n\nJSON Schema:\n{json.dumps(req.schema)}\n\n"
            "Return ONLY valid JSON with keys as in the schema."
        )
        raw = call_llm_chat([{"role":"system","content": system},{"role":"user","content": user}], model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'), temperature=0)
        try:
            data = json.loads(raw)
        except Exception:
            # try to find json block
            start = raw.find("{")
            end = raw.rfind("}")
            data = json.loads(raw[start:end+1]) if start != -1 and end != -1 else {}
        return ExtractResp(data=data, confidence=0.8, spans=[])

    # Fallback: naive regex placeholder — returns empty schema keys.
    data = {k: None for k in req.schema.get("properties", {}).keys()}
    return ExtractResp(data=data, confidence=0.3, spans=[])
