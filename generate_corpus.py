"""
Generates a synthetic enterprise knowledge corpus simulating financial,
consulting, and healthcare research documents (Grey Chain AI's target
industries) for a Retrieval-Augmented Generation (RAG) knowledge assistant.
"""
import json
import random

random.seed(42)

TOPICS = {
    "financial_services": [
        ("Q3 Revenue Growth", "The company reported Q3 revenue growth of 14.2% year-over-year, driven primarily by an increase in net interest income and a 9% rise in fee-based advisory services. Operating margin improved to 31.4% from 28.7% in the prior quarter."),
        ("Credit Risk Provisioning", "The bank increased its provision for credit losses to $420 million this quarter, reflecting a cautious stance amid rising delinquency rates in the unsecured consumer lending portfolio, which grew 60 basis points quarter-over-quarter."),
        ("Digital Banking Adoption", "Digital banking adoption reached 78% of the active customer base, up from 65% two years ago, with mobile transaction volume increasing 22% annually as branch foot traffic declined 11%."),
        ("Regulatory Capital Ratios", "The institution's Common Equity Tier 1 (CET1) ratio stood at 13.1%, comfortably above the regulatory minimum of 10.5%, providing headroom for continued share buybacks and dividend growth."),
        ("Wealth Management AUM", "Assets under management in the wealth management division grew to $184 billion, a 17% increase, aided by net new client inflows of $9.2 billion and favorable market performance."),
    ],
    "healthcare": [
        ("Clinical Trial Enrollment", "Enrollment in the Phase III clinical trial reached 92% of target six weeks ahead of schedule, with the treatment arm showing a 34% reduction in symptom recurrence versus placebo at the 6-month follow-up."),
        ("Hospital Readmission Rates", "30-day hospital readmission rates for congestive heart failure patients declined to 17.8% following the rollout of a post-discharge remote monitoring program, down from 22.3% the prior year."),
        ("Drug Pricing Negotiation", "Under the new pricing framework, the average net price for the top ten specialty drugs decreased 8%, while manufacturer rebates to payers increased, narrowing the gross-to-net gap by 4 percentage points."),
        ("Telehealth Utilization", "Telehealth visit volume stabilized at 19% of total outpatient encounters post-pandemic, with behavioral health accounting for the largest share at 41% of all virtual visits."),
        ("Value-Based Care Contracts", "The health system expanded value-based care contracts to cover 46% of managed lives, achieving a 12% reduction in total cost of care while maintaining quality scores above the 90th percentile."),
    ],
    "consulting": [
        ("Client Engagement Backlog", "The firm's engagement backlog grew to $1.3 billion, a book-to-bill ratio of 1.15x, with digital transformation and GenAI advisory mandates representing the fastest-growing service line at 38% growth."),
        ("Workforce Utilization", "Consultant utilization rates averaged 74% across practice areas, with the technology consulting group exceeding target at 81%, prompting a planned 15% headcount increase in that group next fiscal year."),
        ("M&A Advisory Volume", "M&A advisory deal volume increased 21% year-over-year to 340 completed transactions, with average deal size rising to $210 million amid renewed mid-market dealmaking activity."),
        ("Change Management ROI", "Organizations that paired technology rollouts with structured change management programs saw 2.3x higher user adoption rates and a 30% faster time-to-value compared to technology-only implementations."),
        ("Cost Transformation Program", "The cost transformation program identified $85 million in annualized savings opportunities, with procurement optimization and process automation contributing 60% of the identified value."),
    ],
}

def build_corpus():
    docs = []
    doc_id = 0
    for sector, items in TOPICS.items():
        for title, text in items:
            # create 2 paraphrase-style chunks per doc to simulate real chunked documents
            for variant in range(2):
                doc_id += 1
                docs.append({
                    "doc_id": f"doc_{doc_id:03d}",
                    "sector": sector,
                    "title": title,
                    "text": text if variant == 0 else text + " This trend was highlighted as a key discussion point in the quarterly business review."
                })
    return docs

if __name__ == "__main__":
    corpus = build_corpus()
    with open("corpus.json", "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Generated {len(corpus)} chunks across {len(TOPICS)} sectors")
