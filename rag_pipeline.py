"""
Core RAG retrieval engine: TF-IDF vectorized chunk index + cosine similarity
retriever, with a lightweight extractive answer synthesizer over the
top-k retrieved chunks. Designed to sit behind an API layer (api.py) and
be swapped for a dense embedding index (e.g. FAISS) without changing the
calling contract.
"""
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEngine:
    def __init__(self, corpus_path="corpus.json"):
        with open(corpus_path) as f:
            self.corpus = json.load(f)
        self.texts = [d["text"] for d in self.corpus]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, top_k=3, sector_filter=None):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        ranked = sims.argsort()[::-1]
        results = []
        for idx in ranked:
            doc = self.corpus[idx]
            if sector_filter and doc["sector"] != sector_filter:
                continue
            results.append({**doc, "score": float(sims[idx])})
            if len(results) == top_k:
                break
        return results

    def answer(self, query, top_k=3, sector_filter=None):
        hits = self.retrieve(query, top_k=top_k, sector_filter=sector_filter)
        if not hits or hits[0]["score"] < 0.05:
            return {"answer": "No sufficiently relevant document found in the knowledge base.", "sources": []}
        context = " ".join(h["text"] for h in hits)
        synthesized = f"Based on {len(hits)} retrieved source(s): {hits[0]['text']}"
        return {
            "answer": synthesized,
            "sources": [{"doc_id": h["doc_id"], "title": h["title"], "score": round(h["score"], 3)} for h in hits],
        }


if __name__ == "__main__":
    engine = RAGEngine()
    for q in ["What was the revenue growth this quarter?", "How is telehealth being used?"]:
        print(q, "->", engine.answer(q)["sources"])
