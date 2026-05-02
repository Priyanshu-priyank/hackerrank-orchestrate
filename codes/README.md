# HackerRank Orchestrate — Support Triage Agent

Terminal-based RAG + LLM pipeline that triages support tickets for HackerRank, Claude, and Visa.

---

## Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai) API key

---

## Installation

```bash
cd code
pip install -r requirements.txt
```

---

## Environment Setup

Copy the example and fill in your key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# At least one key required. Add more for automatic rotation on quota errors.
OPENROUTER_API_KEY=your-openrouter-key-here
OPENROUTER_API_KEY_2=
OPENROUTER_API_KEY_3=
```

**Key rotation behaviour:**
1. Agent tries `OPENROUTER_API_KEY` first.
2. On quota / rate-limit error → tries `OPENROUTER_API_KEY_2`, then `_3`, etc.
3. If all keys fail → ticket is safely escalated with an error justification.

---

## How to Run

```bash
# From the code/ directory:

# Run on the real tickets (writes support_tickets/output.csv):
python main.py

# Run on the sample (shows expected vs actual for status and request_type):
python main.py --sample

# Quick smoke test — only process first 5 rows:
python main.py --limit 5
```

Output is written to `../support_tickets/output.csv`.
Sample mode writes to `../support_tickets/output_sample_test.csv` to avoid overwriting.

---

## Expected Output

`support_tickets/output.csv` with columns:

| Column | Values |
|---|---|
| `status` | `replied` / `escalated` |
| `product_area` | string (e.g. `billing`, `screen`, `travel_support`) |
| `response` | user-facing answer, or `""` if escalated |
| `justification` | why this decision was made, grounded in corpus |
| `request_type` | `product_issue` / `feature_request` / `bug` / `invalid` |

---

## Architecture

```
main.py        Reads CSV → orchestrates pipeline → writes output.csv
classifier.py  Pre-LLM: company inference, high-risk keywords, outage detection, injection detection
retriever.py   ChromaDB + sentence-transformers: index corpus once, query per ticket
agent.py       OpenRouter API call with key rotation, retry logic, and structured output parsing
prompts.py     All prompt strings (system prompt + user message builder)
config.py      Constants, paths, keyword lists, key collectors
```

### Pipeline (per ticket)

```
Input row (issue, subject, company)
        │
        ▼
[classifier.py]  Pre-LLM checks:
  • Normalise company (alias → canonical name, or infer from content)
  • Detect empty issue → replied/invalid (no API call)
  • Detect prompt injection → replied/invalid (no API call)
  • Detect system outage → escalated/bug (no API call)
  • Detect high-risk keywords → flag for escalation
        │
        ▼
[retriever.py]  RAG retrieval:
  • Embed issue with all-MiniLM-L6-v2 (local, no API)
  • Query ChromaDB top-3 relevant chunks
  • Filter by company metadata if company is known
  • Fallback to global query if filtered returns 0 results
        │
        ▼
[agent.py]  OpenRouter API call:
  • System prompt: grounding rules + escalation logic + required JSON fields
  • User message: ticket + corpus excerpts + pre-classifier flags
  • JSON mode response, validated and normalised
  • Key rotation + exponential backoff on transient errors
        │
        ▼
Output row (status, product_area, response, justification, request_type)
```

---

## Notes

- **ChromaDB** indexes the corpus on first run (inside `code/chroma_store/`) and reuses it on subsequent runs.
- **No API calls for embeddings** — `all-MiniLM-L6-v2` runs locally via sentence-transformers.
- **No LangChain / LangGraph / live web calls** — pure local RAG pipeline.
- **Deterministic** — `temperature=0` on all LLM calls; corpus index is stable across runs.
