Enterprise Knowledge Assistant — RAG-based Research Q&A

A retrieval-augmented Q&A service that answers natural-language research questions over a multi-sector document corpus (Financial Services, Healthcare, Consulting), with source-cited results — the "Foundation" knowledge-assistant pattern used across GenAI-first enterprise products.

Features
TF-IDF + cosine-similarity retrieval engine (swappable for a dense embedding index like FAISS/pgvector without changing the API contract)
REST API: /query, /sectors, /health
Offline evaluation harness (Precision@k, MRR) wired in as a CI release gate
Dockerized; CI/CD via GitHub Actions
Project Structure
generate_corpus.py   # builds the synthetic multi-sector document corpus
rag_pipeline.py       # RAGEngine: retrieval + extractive answer synthesis
api.py                 # FastAPI service
evaluate.py             # eval harness (Precision@k, MRR) -> eval_report.json
Dockerfile
.github/workflows/ci.yml
PRD.md                 # product requirements + roadmap
Quickstart
bash
pip install -r requirements.txt
python generate_corpus.py
uvicorn api:app --reload

Then query it:

bash
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the revenue growth this quarter?"}'
Run Evaluation
bash
python evaluate.py
Results
30-chunk corpus across 3 sectors
Precision@3 = 0.667, MRR = 1.0 on a 15-query labeled eval set
CI fails the build if MRR drops below 0.85
Docker
bash
docker build -t knowledge-assistant .
docker run -p 8000:8000 knowledge-assistant
Roadmap
v2: dense embedding index for semantic recall
v2: LLM generation step over retrieved context
v3: feedback loop (thumbs up/down) to improve ranking

See PRD.md for full product context and success metrics.
