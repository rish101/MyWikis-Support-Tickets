# WikiChat Support Data Pipeline

Scripts I built during my internship at MyWikis, contributing to **WikiChat**, an AI-powered customer-support chatbot. My work focused on the data/API side of the system: extracting support-ticket data from WHMCS and preparing it for semantic retrieval via Pinecone.

## Architecture

```
WHMCS (ticket system)
        │
        ▼
WHMCS API  →  whmcs_export.py
        │
        ▼
Excel export (support_tickets_full.xlsx)
        │
        ▼
Local embedding model (sentence-transformers/all-MiniLM-L6-v2)  →  pinecone_upload.py
        │
        ▼
Pinecone vector database
        │
        ▼
Relevant support context retrieved by WikiChat
```

## Scripts

### `whmcs_export.py`
Pulls support tickets from the WHMCS API and exports them to Excel.
- Paginates through `GetTickets` to retrieve all ticket records
- Fetches full details (including message history) per ticket via `GetTicket`, using a thread pool for speed
- Handles retries for transient API/network errors
- Outputs a structured `support_tickets_full.xlsx`

### `pinecone_upload.py`
Embeds the exported ticket data and upserts it into Pinecone.
- Loads ticket data from the Excel export
- Generates embeddings locally using `sentence-transformers/all-MiniLM-L6-v2` (via `transformers` + `torch`)
- Creates the Pinecone index if it doesn't exist (384-dim, cosine similarity, serverless)
- Upserts vectors in batches, with ticket subject/client/priority/status/message stored as metadata

## Setup

```bash
pip install requests pandas openpyxl pinecone-client torch transformers
```

Set the following environment variables before running either script:

```bash
export WHMCS_API_URL=https://your-panel.example.com/includes/api.php
export WHMCS_API_IDENTIFIER=your_whmcs_api_identifier
export WHMCS_API_SECRET=your_whmcs_api_secret

export PINECONE_API_KEY=your_pinecone_api_key
export PINECONE_HOST=your_pinecone_index_host
```

## Usage

```bash
python whmcs_export.py     # WHMCS -> support_tickets_full.xlsx
python pinecone_upload.py  # support_tickets_full.xlsx -> Pinecone index
```

## Context

This was one part of a larger effort supporting WikiChat's development — I contributed to the data pipeline and retrieval layer, not the entire chatbot system end-to-end. This repo captures the API automation, data processing, and vector-retrieval setup portion of that work.
