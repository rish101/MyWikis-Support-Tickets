# MyWikis Support Ticket RAG

A retrieval-augmented support chatbot built over a MediaWiki hosting company's
historical support tickets. Past tickets are exported from WHMCS, embedded, and
indexed, then a local language model answers new questions using only the
retrieved tickets as context.

The retrieval half was built during an internship at MyWikis. The generation
half, the evaluation harness, and the local index were added afterwards as a
personal continuation of that work.

## Pipeline

```
WHMCS API  ->  whmcs_export.py    ->  Excel export
Excel      ->  build_index.py     ->  local vector index   (or pinecone_script.py -> Pinecone)
Question   ->  chatbot.py         ->  grounded answer + cited ticket IDs
```

## Running it

The real ticket export is company data and is not in this repo. A synthetic
corpus of 30 realistic MediaWiki hosting tickets stands in for it so the whole
pipeline runs with no API keys.

```bash
pip install torch transformers numpy pandas openpyxl

python make_synthetic_tickets.py     # writes data/synthetic_tickets.json
python build_index.py                # embeds the corpus into index/
python chatbot.py --question "How do I increase the upload size limit?"
python evaluate.py                   # retrieval metrics
python -m pytest tests/ -v           # unit tests, no model weights needed
```

To run against a real WHMCS export instead:

```bash
python whmcs_export.py               # needs WHMCS_API_* env vars
python build_index.py --input support_tickets_full.xlsx
```

## Files

| File | Purpose |
|---|---|
| `whmcs_export.py` | Pulls tickets from the WHMCS API. Paginated, with a thread pool and retries. |
| `embeddings.py` | Shared embedding model wrapper and attention-mask-weighted mean pooling. |
| `pinecone_script.py` | Embeds an export and upserts to a Pinecone serverless index. |
| `vector_store.py` | Local brute-force cosine store, same shape as the Pinecone path. |
| `build_index.py` | Embeds a corpus and writes a local index. |
| `chatbot.py` | Retrieval, prompt construction, generation, abstention. |
| `evaluate.py` | Retrieval metrics and score-separation analysis. |
| `make_synthetic_tickets.py` | Generates the stand-in corpus. |
| `tests/test_pipeline.py` | Unit tests that run without downloading model weights. |

## Design notes

**Pooling.** `all-MiniLM-L6-v2` needs attention-mask-weighted mean pooling. The
original inline implementation padded every input to 512 tokens and then took a
plain mean over the full sequence, which averaged several hundred padding vectors
into every short ticket. `embeddings.mean_pool` fixes this and
`tests/test_pipeline.py` pins the behaviour against the naive version.

**Abstention.** If the best retrieval score is below `MIN_SCORE`, the generator is
never called and the bot says it does not know. A support bot that confidently
invents a configuration setting is worse than one that declines. `evaluate.py`
reports the score distribution for in-scope and out-of-scope questions so the
threshold can be set from data rather than guessed.

**Testability.** Prompt construction is separate from generation, and the
retriever takes an injected embedder, so the pipeline logic can be tested with a
stub in under a second instead of loading model weights.

## Known issues

- `whmcs_export.py` cannot distinguish "no more pages" from "this request failed",
  so a network error mid-export silently ends the loop and produces a partial
  file that still reports success.
- The local store is a brute-force matmul. Exact and instant at this scale, but it
  would need a real ANN index past roughly a hundred thousand tickets.
