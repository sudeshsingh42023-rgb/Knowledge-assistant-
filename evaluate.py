"""
Evaluation harness: measures retrieval quality (Precision@k, MRR) against
a hand-labeled set of queries with known relevant document IDs. This is
the kind of offline eval a PM would use to gate releases in CI/CD before
a retrieval-index change ships to production.
"""
import json
from rag_pipeline import RAGEngine

EVAL_SET = [
    {"query": "What was quarterly revenue growth driven by?", "relevant_titles": ["Q3 Revenue Growth"]},
    {"query": "How much did the bank set aside for credit losses?", "relevant_titles": ["Credit Risk Provisioning"]},
    {"query": "What percentage of customers use mobile banking?", "relevant_titles": ["Digital Banking Adoption"]},
    {"query": "What is the bank's capital ratio versus the regulatory minimum?", "relevant_titles": ["Regulatory Capital Ratios"]},
    {"query": "How much did wealth management assets grow?", "relevant_titles": ["Wealth Management AUM"]},
    {"query": "What was the outcome of the Phase III clinical trial?", "relevant_titles": ["Clinical Trial Enrollment"]},
    {"query": "Did the remote monitoring program reduce hospital readmissions?", "relevant_titles": ["Hospital Readmission Rates"]},
    {"query": "How did drug pricing negotiations affect net prices?", "relevant_titles": ["Drug Pricing Negotiation"]},
    {"query": "What share of visits are now telehealth?", "relevant_titles": ["Telehealth Utilization"]},
    {"query": "How much did value-based care contracts reduce cost of care?", "relevant_titles": ["Value-Based Care Contracts"]},
    {"query": "How big is the consulting firm's engagement backlog?", "relevant_titles": ["Client Engagement Backlog"]},
    {"query": "What is the consultant utilization rate?", "relevant_titles": ["Workforce Utilization"]},
    {"query": "How much did M&A advisory deal volume increase?", "relevant_titles": ["M&A Advisory Volume"]},
    {"query": "Does change management improve technology adoption?", "relevant_titles": ["Change Management ROI"]},
    {"query": "How much cost savings did the transformation program find?", "relevant_titles": ["Cost Transformation Program"]},
]


def evaluate(top_k=3):
    engine = RAGEngine()
    precisions, reciprocal_ranks = [], []
    for item in EVAL_SET:
        hits = engine.retrieve(item["query"], top_k=top_k)
        hit_titles = [h["title"] for h in hits]
        relevant_hits = [t for t in hit_titles if t in item["relevant_titles"]]
        precisions.append(len(relevant_hits) / top_k)
        rr = 0
        for rank, t in enumerate(hit_titles, start=1):
            if t in item["relevant_titles"]:
                rr = 1 / rank
                break
        reciprocal_ranks.append(rr)

    results = {
        "num_queries": len(EVAL_SET),
        "top_k": top_k,
        "precision_at_k": round(sum(precisions) / len(precisions), 3),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
    }
    return results


if __name__ == "__main__":
    results = evaluate(top_k=3)
    print(json.dumps(results, indent=2))
    with open("eval_report.json", "w") as f:
        json.dump(results, f, indent=2)
