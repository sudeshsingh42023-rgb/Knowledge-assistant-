# PRD: Enterprise Knowledge Assistant (RAG-based Research Q&A)

## Problem
Analysts across Financial Services, Healthcare, and Consulting client accounts
spend significant time manually searching scattered reports to answer
recurring research questions (e.g., "what was the margin trend," "what were
trial enrollment numbers"). This slows down client deliverables and creates
inconsistent answers across teams.

## Goal
Ship a GenAI-first knowledge assistant that retrieves the most relevant
source passages for a natural-language question and returns a grounded,
source-cited answer — the foundation-layer pattern Grey Chain AI's
"Foundation" product uses across its industry verticals.

## Success Metrics
- Retrieval quality: MRR ≥ 0.85 on the labeled eval set (gates every CI build)
- P95 API latency < 300ms for retrieval (excludes any generative LLM call)
- Sector-scoped queries return zero cross-sector leakage

## Scope (v1)
- TF-IDF/cosine retrieval over chunked documents (swappable for a dense
  embedding index without changing the API contract)
- REST API (`/query`, `/sectors`, `/health`) for downstream product integration
- Offline evaluation harness wired into CI as a release gate
- Containerized deployment (Docker) + CI/CD pipeline (GitHub Actions)

## Out of Scope (v1) / Roadmap
- v2: swap TF-IDF for a dense embedding index (FAISS/pgvector) for semantic recall
- v2: add an LLM generation step over retrieved context for abstractive answers
- v3: user feedback loop (thumbs up/down) to continuously improve ranking

## Results (this build)
- 30-chunk multi-sector corpus (Financial Services, Healthcare, Consulting)
- Precision@3 = 0.667, MRR = 1.0 on a 15-query labeled eval set
- CI pipeline fails the build automatically if MRR drops below 0.85
