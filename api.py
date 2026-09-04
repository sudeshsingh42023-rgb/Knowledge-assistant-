"""
Production-facing API layer for the Enterprise Knowledge Assistant.
Exposes /query for RAG-based Q&A and /health for deployment liveness checks.
"""
from fastapi import FastAPI, Query
from pydantic import BaseModel
from rag_pipeline import RAGEngine

app = FastAPI(title="Enterprise Knowledge Assistant API", version="1.0.0")
engine = RAGEngine()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    sector: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "corpus_size": len(engine.corpus)}


@app.post("/query")
def query(req: QueryRequest):
    return engine.answer(req.question, top_k=req.top_k, sector_filter=req.sector)


@app.get("/sectors")
def sectors():
    return sorted(set(d["sector"] for d in engine.corpus))
