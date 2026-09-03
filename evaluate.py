"""
Evaluation harness for the retrieval layer.

Retrieval quality decides whether the generated answer can possibly be right, so it
is measured directly rather than by reading chatbot output and guessing. Each test
question is a paraphrase, never a copy of a ticket subject, so the score reflects
semantic matching rather than string overlap.

Reported metrics:
  recall@1 / recall@3  how often the correct ticket is the top hit, or in the top 3
  MRR                  mean reciprocal rank of the correct ticket
  score separation     top-1 score distribution for in-scope vs out-of-scope questions,
                       which is what MIN_SCORE in chatbot.py should be tuned against

Usage:
    python evaluate.py
    python evaluate.py --index index/ --top-k 3

Requires:
    pip install torch transformers numpy
"""

from __future__ import annotations

import argparse
import statistics

from embeddings import Embedder
from vector_store import LocalVectorStore

# (question, expected ticket id). Questions are deliberately worded differently
# from the ticket subjects they should match.
IN_SCOPE = [
    ("Our editor keeps falling back to wikitext instead of the visual one", "1000"),
    ("Can we upload PDFs bigger than the current limit?", "1001"),
    ("The browser says our site is not secure, something about a certificate", "1002"),
    ("We want structured data tables built from templates, which extension?", "1003"),
    ("Junk pages keep appearing overnight from new accounts", "1004"),
    ("How can I get all our content out including revision history?", "1005"),
    ("Can our staff sign in with their work Google accounts?", "1006"),
    ("Everything takes forever to load, what is slowing the site down?", "1007"),
    ("Someone removed a page by accident, can we get it back?", "1008"),
    ("Nobody on our team is receiving watchlist emails", "1009"),
    ("We run our own wiki elsewhere and want to move it to you", "1010"),
    ("How do I put our company logo in the corner of the wiki?", "1011"),
    ("Saving edits throws a lock wait timeout error", "1012"),
    ("How do I give one of our users admin rights?", "1013"),
    ("How long do you keep backups for?", "1014"),
    ("Images broke after we mass renamed a lot of files", "1015"),
    ("Do you support SMW and which plan includes it?", "1016"),
    ("We seem to have been billed twice this month", "1017"),
    ("How do we close our account and take our data with us?", "1018"),
    ("We only want our own employees to be able to read the wiki", "1019"),
    ("How does moving to a newer MediaWiki release work?", "1020"),
    ("Pages we just created are not appearing in search results", "1021"),
    ("Can we use pywikibot to make bulk edits through the API?", "1022"),
    ("Our domain still will not load after we updated DNS", "1023"),
    ("Can we require two factor authentication for administrators?", "1024"),
    ("There is an unfamiliar IP address in our page history", "1025"),
    ("Infoboxes are showing raw wikitext since the last update", "1026"),
    ("Is there a cap on how many accounts the cheapest plan allows?", "1027"),
    ("How do I turn on slash subpages in the main namespace?", "1028"),
    ("Can we get a test copy of our wiki to try template changes?", "1029"),
]

# Questions the corpus cannot answer. The bot should abstain rather than guess.
OUT_OF_SCOPE = [
    "What is the capital of France?",
    "How do I reset my Netflix password?",
    "Write me a Python function that reverses a string",
    "What is the weather forecast for tomorrow?",
    "Who won the World Cup in 2018?",
]


def evaluate(index_dir: str, top_k: int) -> dict:
    store = LocalVectorStore.load(index_dir)
    embedder = Embedder()

    questions = [q for q, _ in IN_SCOPE]
    vectors = embedder.encode(questions)

    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks = []
    in_scope_scores = []
    misses = []

    for (question, expected), vector in zip(IN_SCOPE, vectors):
        results = store.query(vector, top_k=top_k)
        ids = [r["ticket_id"] for r in results]
        in_scope_scores.append(results[0]["score"])

        if ids and ids[0] == expected:
            hits_at_1 += 1
        if expected in ids:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / (ids.index(expected) + 1))
        else:
            reciprocal_ranks.append(0.0)
            misses.append(
                {
                    "question": question,
                    "expected": expected,
                    "got": [(r["ticket_id"], r["subject"], round(r["score"], 3)) for r in results],
                }
            )

    oos_vectors = embedder.encode(OUT_OF_SCOPE)
    oos_scores = [store.query(v, top_k=1)[0]["score"] for v in oos_vectors]

    total = len(IN_SCOPE)
    return {
        "total": total,
        "recall_at_1": hits_at_1 / total,
        "recall_at_k": hits_at_k / total,
        "mrr": sum(reciprocal_ranks) / total,
        "in_scope_scores": in_scope_scores,
        "oos_scores": oos_scores,
        "misses": misses,
        "top_k": top_k,
    }


def report(r: dict) -> None:
    print("\n=== Retrieval evaluation ===")
    print(f"questions:    {r['total']}")
    print(f"recall@1:     {r['recall_at_1']:.1%}")
    print(f"recall@{r['top_k']}:     {r['recall_at_k']:.1%}")
    print(f"MRR:          {r['mrr']:.3f}")

    in_lo = min(r["in_scope_scores"])
    in_mean = statistics.mean(r["in_scope_scores"])
    oos_hi = max(r["oos_scores"])
    oos_mean = statistics.mean(r["oos_scores"])

    print("\n=== Score separation ===")
    print(f"in-scope  top-1 score: min {in_lo:.3f}  mean {in_mean:.3f}")
    print(f"out-of-scope top-1:    max {oos_hi:.3f}  mean {oos_mean:.3f}")

    if in_lo > oos_hi:
        midpoint = (in_lo + oos_hi) / 2
        print(f"\nClean separation. A MIN_SCORE around {midpoint:.2f} rejects every")
        print("out-of-scope question while keeping every in-scope one.")
    else:
        print(f"\nOverlap: the weakest in-scope question ({in_lo:.3f}) scores at or below")
        print(f"the strongest out-of-scope one ({oos_hi:.3f}), so no single threshold")
        print("separates them cleanly. Any MIN_SCORE trades false abstentions against")
        print("answering questions the corpus cannot support.")

    if r["misses"]:
        print(f"\n=== Misses ({len(r['misses'])}) ===")
        for m in r["misses"]:
            print(f"\nQ: {m['question']}")
            print(f"   expected ticket {m['expected']}, retrieved:")
            for tid, subject, score in m["got"]:
                print(f"     [{tid}] {subject}  ({score:.3f})")
    else:
        print("\nNo misses.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--index", default="index", help="index directory")
    parser.add_argument("--top-k", type=int, default=3, help="retrieval depth for recall@k")
    args = parser.parse_args()
    report(evaluate(args.index, args.top_k))


if __name__ == "__main__":
    main()
