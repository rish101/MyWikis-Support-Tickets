"""
Embeds a ticket corpus and writes a local vector index.

This is the local counterpart to pinecone_script.py. It reads either the synthetic
corpus (default) or a real WHMCS export produced by whmcs_export.py, embeds each
ticket, and saves a LocalVectorStore that chatbot.py can query.

Usage:
    python build_index.py
    python build_index.py --input support_tickets_full.xlsx --output index/

Requires:
    pip install torch transformers numpy pandas openpyxl
"""

from __future__ import annotations

import argparse
import json
import os

from embeddings import Embedder, ticket_text
from vector_store import LocalVectorStore

DEFAULT_INPUT = os.path.join("data", "synthetic_tickets.json")
DEFAULT_OUTPUT = "index"


def load_tickets(path: str) -> list[dict]:
    """Load tickets from JSON or from the Excel file whmcs_export.py writes."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    if path.endswith((".xlsx", ".xls")):
        import pandas as pd

        df = pd.read_excel(path)
        return df.to_dict(orient="records")

    raise ValueError(f"Unsupported input format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local vector index of support tickets.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="ticket corpus (.json or .xlsx)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="directory to write the index to")
    args = parser.parse_args()

    tickets = load_tickets(args.input)
    print(f"Loaded {len(tickets)} tickets from {args.input}")

    embedder = Embedder()
    texts = [ticket_text(t) for t in tickets]
    vectors = embedder.encode(texts)
    print(f"Embedded {len(vectors)} tickets into {vectors.shape[1]}-dim vectors")

    metadata = [
        {
            "ticket_id": str(t.get("Ticket ID", i)),
            "subject": t.get("Subject", ""),
            "department": t.get("Department Name", ""),
            "priority": t.get("Priority", ""),
            "status": t.get("Status", ""),
            "messages": t.get("Messages", ""),
        }
        for i, t in enumerate(tickets)
    ]

    store = LocalVectorStore()
    store.upsert(vectors, metadata)
    store.save(args.output)


if __name__ == "__main__":
    main()
