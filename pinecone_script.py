"""
Uploads processed WHMCS support-ticket data to Pinecone as embeddings
for semantic retrieval by the WikiChat support chatbot.

Pipeline: WHMCS ticket export (Excel) -> local embedding model -> Pinecone vector index

Requires:
    pip install pandas pinecone-client torch transformers openpyxl

Environment:
    PINECONE_API_KEY   - your Pinecone API key
    PINECONE_HOST      - your Pinecone index host URL
"""

import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
import torch
from transformers import AutoTokenizer, AutoModel

# --- Config: loaded from environment, never hardcoded ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_HOST = os.environ.get("PINECONE_HOST")

if not PINECONE_API_KEY or not PINECONE_HOST:
    raise EnvironmentError(
        "Set PINECONE_API_KEY and PINECONE_HOST environment variables before running. "
        "e.g. export PINECONE_API_KEY=your_key_here"
    )

INDEX_NAME = "support-tickets"
EMBEDDING_DIMENSION = 384  # matches the sentence-transformers model output
INPUT_FILE = "support_tickets_full.xlsx"
BATCH_SIZE = 50

# --- Step 1: Initialize Pinecone client ---
print("Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
print("Initialized Pinecone successfully.")

# --- Step 2: Create the index if it doesn't already exist ---
existing_indexes = pc.list_indexes().names()
print("Existing indexes:", existing_indexes)

if INDEX_NAME not in existing_indexes:
    print(f"Creating index '{INDEX_NAME}' with dimension {EMBEDDING_DIMENSION}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"Index '{INDEX_NAME}' already exists.")

# --- Step 3: Connect to the index ---
try:
    index = pc.Index(INDEX_NAME, host=PINECONE_HOST)
    print(f"Successfully connected to the index '{INDEX_NAME}'")
except Exception as e:
    raise RuntimeError(f"Failed to connect to the index '{INDEX_NAME}': {e}")

# --- Step 4: Load embedding model ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print("Loading transformer model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
print("Model and tokenizer loaded successfully.")


def embed_text(text: str) -> list[float]:
    """Generate a mean-pooled sentence embedding for a given text."""
    tokens = tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=512
    )
    with torch.no_grad():
        model_output = model(**tokens)
    embedding = torch.mean(model_output.last_hidden_state, dim=1).squeeze().tolist()
    return embedding


# --- Step 5: Load ticket data ---
df = pd.read_excel(INPUT_FILE)
print(f"Loaded {len(df)} tickets from {INPUT_FILE}.")

# --- Step 6: Embed and upsert in batches ---
vectors_to_upsert = []

for idx, row in df.iterrows():
    ticket_id = str(row["Ticket ID"])
    combined_text = f"Subject: {row['Subject']}. Messages: {row['Messages']}"
    embedding = embed_text(combined_text)

    vectors_to_upsert.append(
        {
            "id": ticket_id,
            "values": embedding,
            "metadata": {
                "subject": row["Subject"],
                "client_name": row["Client Name"],
                "priority": row["Priority"],
                "status": row["Status"],
                "message": row["Messages"],
            },
        }
    )

    if len(vectors_to_upsert) >= BATCH_SIZE:
        try:
            print(f"Upserting a batch of {len(vectors_to_upsert)} vectors...")
            index.upsert(vectors=vectors_to_upsert)
            print(f"Successfully upserted {len(vectors_to_upsert)} vectors.")
        except Exception as e:
            print(f"Failed to upsert batch of vectors: {e}")
        vectors_to_upsert = []

if vectors_to_upsert:
    try:
        print(f"Upserting the remaining {len(vectors_to_upsert)} vectors...")
        index.upsert(vectors=vectors_to_upsert)
        print(f"Successfully upserted {len(vectors_to_upsert)} remaining vectors.")
    except Exception as e:
        print(f"Failed to upsert remaining batch of vectors: {e}")

print("All ticket data uploaded to Pinecone successfully.")
