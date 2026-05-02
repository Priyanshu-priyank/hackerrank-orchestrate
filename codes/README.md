# HackerRank Orchestrate — Support Triage Agent

Terminal-based RAG + LLM pipeline that triages support tickets for HackerRank, Claude, and Visa.

---

## Prerequisites

- Python 3.11+
- Gemini API key (primary) and/or Anthropic API key (fallback)

---

## Installation

```bash
cd code
pip install -r requirements.txt
```

---

## Environment Setup

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# ── Gemini (primary LLM) ──────────────────────────────────────────────────────
# At least one required. Add more for automatic key rotation on quota errors.
GEMINI_API_KEY=your-gemini-key-1
GEMINI_API_KEY_2=your-gemini-key-2
GEMINI_API_KEY_3=your-gemini-key-3

# ── Anthropic (fallback LLM) ──────────────────────────────────────────────────
# Used automatically if all Gemini keys are exhausted.
ANTHROPIC_API_KEY=your-anthropic-key
ANTHROPIC_API_KEY_2=your-second-anthropic-key
```

**Key rotation behaviour:**
1. Agent tries `GEMINI_API_KEY` first.
2. On quota / rate-limit error → tries `GEMINI_API_KEY_2`, then `_3`, etc.
3. If all Gemini keys fail → falls back to `ANTHROPIC_API_KEY`, then `ANTHROPIC_API_KEY_2`, etc.
4. If everything fails → ticket is escalated with an error justification (safe default).

You can use Gemini-only, Anthropic-only, or both. Any keys left blank are skipped.

---

## How to Run

```bash
# From the code/ directory:

# Run on the real tickets:
python main.py

# Run on the sample (shows expected vs actual for discrete fields):
python main.py --sample

# Quick smoke test — only process first 5 rows:
python main.py --limit 5
```

Output is written to `../support_tickets/output.csv`.
(Sample mode writes to `../support_tickets/output_sample_test.csv` to avoid overwriting.)

---

## Expected Output

`support_tickets/output.csv` with columns:

| Column | Values |
|---|---|
| `status` | `replied` / `escalated` |
| `product_area` | string (e.g. `billing`, `screen`, `travel_support`) |
| `response` | user-facing answer, or `""` if escalated |
| `justification` | why this decision was made |
| `request_type` | `product_issue` / `feature_request` / `bug` / `invalid` |

---

## Architecture

```
main.py        Reads CSV → orchestrates pipeline → writes output.csv
classifier.py  Pre-LLM: company inference, high-risk keywords, injection detection
retriever.py   ChromaDB + sentence-transformers: index corpus once, query per ticket
agent.py       LLM call: Gemini primary → Anthropic fallback, with key rotation
prompts.py     All prompt strings (system prompt + user message builder)
config.py      Constants, paths, key collectors
```

---

## Notes

- **ChromaDB** indexes the corpus on first run and reuses it on subsequent runs.
- **No API calls for embeddings** — `all-MiniLM-L6-v2` runs locally.
- **No LangChain / LangGraph / live web calls** — pure local RAG pipeline.
- **Ollama / local models**: not included. Local models require a running Ollama server
  and produce unstructured output that needs extra parsing. Given you already have
  Gemini + Anthropic fallback, Ollama adds complexity without reliability benefit
  for a 24-hour hackathon. Add it post-submission if desired.
