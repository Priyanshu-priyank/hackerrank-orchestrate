# requirements.md — HackerRank Orchestrate: Multi-Domain Support Triage Agent

> **Purpose:** This file is the complete technical specification for the support triage agent.
> Paste this into any AI coding assistant to get full context instantly.
> Last updated: May 2026.

---

## 1. Project Summary

Build a **terminal-based Python agent** that reads support tickets from a CSV, processes each through a RAG + LLM pipeline, and writes back a structured CSV with 5 output columns.

- **Hackathon:** HackerRank Orchestrate, May 1–2, 2026
- **Deadline:** May 2, 2026, 11:00 AM IST
- **Language:** Python 3.11+
- **LLM:** Gemini API (`gemini-2.5-flash`) via Google Gen AI SDK
- **Retrieval:** ChromaDB (local vector DB) + sentence-transformers (local embeddings)
- **No LangChain. No LangGraph. No live web calls.**

---

## 2. Given Repository Structure (Do NOT change this layout)

```
hackerrank-orchestrate-may26/
├── AGENTS.md                        # Hackathon rules for AI tools (read-only)
├── README.md                        # Quickstart
├── problem_statement.md             # Full task spec
├── .env.example                     # Copy to .env for secrets
├── .gitignore
├── code/                            # ← ALL YOUR CODE GOES HERE
│   └── main.py                      # Entry point (provided empty)
├── data/                            # Local support corpus (READ ONLY, do not modify)
│   ├── hackerrank/                  # HackerRank help center docs
│   ├── claude/                      # Claude Help Center docs
│   └── visa/                        # Visa consumer support docs
└── support_tickets/
    ├── sample_support_tickets.csv   # Input + expected output (use for testing)
    ├── support_tickets.csv          # Input only (run agent on this)
    └── output.csv                   # ← Write agent predictions here
```

---

## 3. Code Structure to Build (inside `code/`)

```
code/
├── main.py            # Entry point: reads CSV, orchestrates pipeline, writes output.csv
├── retriever.py       # ChromaDB indexing + query (the RAG layer)
├── agent.py           # Gemini API call + structured output parsing (the brain)
├── classifier.py      # Pre-LLM risk detection + company inference
├── prompts.py         # All prompt templates in one place (no prompts in other files)
├── config.py          # Constants: model name, top_k, thresholds, paths
├── requirements.txt   # Pinned dependencies
└── README.md          # How to install and run (mandatory for evaluation)
```

---

## 4. Input Schema

File: `support_tickets/support_tickets.csv`

| Field | Type | Notes |
|---|---|---|
| `issue` | string | Main ticket body. May contain multiple requests, irrelevant text, or even malicious text |
| `subject` | string | May be blank, partial, noisy, or misleading |
| `company` | string | `HackerRank`, `Claude`, `Visa`, or `None` |

**Edge cases that MUST be handled:**
- `company = "None"` → infer from content
- Multiple requests in one issue → handle the primary one, note others
- Irrelevant/malicious text injected into the issue → detect and classify as `invalid`
- Empty or nonsense subject → ignore subject, rely on issue content only
- Prompt injection attempts (e.g., "ignore previous instructions") → treat as `invalid`, escalate

---

## 5. Required Output Schema

File: `support_tickets/output.csv`

| Column | Allowed Values | Description |
|---|---|---|
| `status` | `replied`, `escalated` | Whether agent answered or routed to human |
| `product_area` | free string (e.g., `billing`, `account`, `screen`, `general_support`) | Most relevant support category |
| `response` | free string | User-facing answer grounded in corpus. Empty if escalated |
| `justification` | free string | Why this decision was made (traceable to corpus) |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` | Best-fit classification |

---

## 6. Escalation Rules (HARD RULES — Non-Negotiable)

**Always escalate when:**
- Fraud, unauthorized transactions, or stolen card/account
- Account compromise or security breach
- Legal threats or mentions of lawsuits
- Billing disputes requiring account verification
- Password reset or account deletion requiring identity verification
- The support corpus has NO relevant documentation for the issue
- The issue is ambiguous AND could cause harm if answered incorrectly
- System is down / critical outage affecting multiple users

**Always reply (do not escalate) when:**
- Clear FAQ answered directly in the corpus
- How-to questions with documented steps
- Feature questions with documented behavior
- Out-of-scope / invalid questions → reply with "this is out of scope" message
- General informational questions covered by corpus

**Gray area → escalate.** When in doubt, escalate. Never guess policies.

---

## 7. Full Pipeline (Per Ticket)

```
Input row (issue, subject, company)
        │
        ▼
[classifier.py] ─ Pre-LLM checks:
  • Detect company if "None" (keyword/embedding match)
  • Detect high-risk keywords → flag for escalation
  • Detect prompt injection / irrelevant content → flag as invalid
        │
        ▼
[retriever.py] ─ RAG retrieval:
  • Embed the issue using sentence-transformers (all-MiniLM-L6-v2)
  • Query ChromaDB for top-3 relevant chunks
  • Filter by company metadata if company is known
  • Return chunks + metadata (source file, company, product_area)
        │
        ▼
[agent.py] ─ Gemini API call:
  • System prompt: triage rules + escalation logic
  • User message: issue + subject + company + retrieved chunks
  • Force structured JSON output via tool use
  • Parse and validate response
        │
        ▼
Output row (status, product_area, response, justification, request_type)
```

---

## 8. RAG Implementation Details

### 8.1 Chunking Strategy

Chunk at startup when indexing the corpus. Use paragraph-based chunking (split on `\n\n`), max 500 tokens per chunk. If a paragraph exceeds 500 tokens, split on sentence boundaries.

```python
def _chunk_text(self, text: str, max_tokens: int = 500) -> list[str]:
    paragraphs = text.split('\n\n')
    chunks = []
    current = ""
    for para in paragraphs:
        combined_len = len(current.split()) + len(para.split())
        if combined_len > max_tokens and current:
            chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).strip()
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c.strip()) > 30]  # skip tiny chunks
```

### 8.2 Embedding Model

Use `sentence-transformers/all-MiniLM-L6-v2`. This is local, no API calls, 384-dimensional vectors, fast inference.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("text here")  # returns numpy array
```

### 8.3 ChromaDB Setup

```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(
    name="support_corpus",
    metadata={"hnsw:space": "cosine"}
)
```

Use `PersistentClient` so the corpus is only indexed once (not re-indexed every run). Check if collection already has documents before re-indexing.

### 8.4 Metadata per Chunk

```python
{
    "company": "hackerrank",   # or "claude" or "visa"
    "filename": "billing_faq.md",
    "product_area": "billing",
    "chunk_index": 3
}
```

### 8.5 Query

```python
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3,
    where={"company": "hackerrank"},  # filter by company if known
    include=["documents", "metadatas", "distances"]
)
```

If company is `None` or unknown, query without the `where` filter.

---

## 9. Gemini API Call

### 9.1 Model

Always use: `gemini-2.5-flash`

### 9.2 Structured Output via Response Schema

Force structured output with Gemini's JSON response mode and a Pydantic response schema.

### 9.3 API Call Pattern

```python
import os
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

from config import MODEL
from prompts import SYSTEM_PROMPT


class TriageDecision(BaseModel):
    status: Literal["replied", "escalated"]
    product_area: str
    response: str
    justification: str
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"]


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model=MODEL,
    contents=user_message,
    config=types.GenerateContentConfig(
        max_output_tokens=1024,
        temperature=0,
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=TriageDecision,
    ),
)

result = response.parsed
```

### 9.4 System Prompt (in `prompts.py`)

```
You are a support triage agent for three products: HackerRank, Claude, and Visa.

Your job for each ticket:
1. Identify the type of request
2. Classify into a product area
3. Assess urgency and risk
4. Decide: reply or escalate
5. Retrieve grounding from the provided corpus excerpts
6. Generate a safe, accurate response

STRICT RULES:
- Base ALL responses ONLY on the provided corpus excerpts. Never use outside knowledge.
- If the corpus does not cover the issue, escalate. Do not guess.
- Escalate immediately for: fraud, account compromise, stolen cards, security breaches,
  billing disputes needing identity verification, legal threats, system outages.
- For out-of-scope or irrelevant questions, reply with a polite "out of scope" message,
  set status=replied and request_type=invalid.
- Never fabricate policies, steps, or contact numbers not present in the corpus.
- If the company is unknown, infer from the issue content.
- Be concise. Response should be 2-6 sentences for simple issues, bullet steps for how-to.

ESCALATION FORMAT:
- response: empty string ""
- justification: explain why escalation was needed

REPLY FORMAT:
- response: helpful user-facing answer grounded in corpus
- justification: which corpus section answered this and why
```

---

## 10. Config (`config.py`)

```python
MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 3
MAX_CHUNK_TOKENS = 500
CORPUS_PATH = "../data"
CHROMA_PERSIST_PATH = "./chroma_store"
INPUT_CSV = "../support_tickets/support_tickets.csv"
OUTPUT_CSV = "../support_tickets/output.csv"
SAMPLE_CSV = "../support_tickets/sample_support_tickets.csv"

HIGH_RISK_KEYWORDS = [
    "fraud", "unauthorized", "stolen", "hacked", "compromised",
    "lawsuit", "legal action", "billing dispute", "charge back",
    "account deleted", "security breach", "identity theft"
]
```

---

## 11. Main Entry Point (`main.py`)

```python
# Pseudocode structure
def main():
    retriever = Retriever(CORPUS_PATH)   # indexes corpus at startup (once)
    rows = read_csv(INPUT_CSV)
    results = []
    for row in rows:
        classification = classify(row)       # pre-LLM checks
        chunks = retriever.query(row.issue, company=row.company)
        result = run_agent(row, chunks, classification)
        results.append(result)
        print(f"[{row.index}] {result.status} | {result.request_type} | {result.product_area}")
    write_csv(OUTPUT_CSV, results)

if __name__ == "__main__":
    main()
```

---

## 12. Determinism Requirements

- Pin all dependency versions in `requirements.txt`
- Pass `seed` to any random operations
- ChromaDB's HNSW search has some non-determinism at high load; acceptable for this task
- Gemini API: use `temperature=0` for more deterministic structured outputs.

---

## 13. Required `requirements.txt`

```
google-genai==1.74.0
chromadb==0.6.3
sentence-transformers==3.4.1
pandas==2.2.3
python-dotenv==1.0.1
pydantic==2.11.3
```

---

## 14. Required `code/README.md` Contents

Must include:
1. Prerequisites (Python 3.11+)
2. Installation steps (`pip install -r requirements.txt`)
3. Environment setup (`cp .env.example .env`, add `GEMINI_API_KEY`)
4. How to run (`python main.py`)
5. Expected output location (`support_tickets/output.csv`)
6. How to test against sample (`python main.py --sample`)

---

## 15. Sample Data Analysis (from `sample_support_tickets.csv`)

Observed patterns from the provided sample data:

| Pattern | Example | Expected Behavior |
|---|---|---|
| Clear FAQ | "How long do tests stay active?" | `replied`, `product_issue` |
| Site outage | "site is down" | `escalated`, `bug` |
| How-to with steps | "How to add extra time for candidates" | `replied`, `product_issue` |
| Account deletion | "please delete my account" | `replied`, `product_issue` |
| Privacy concern | "delete a Claude conversation" | `replied`, `product_issue` |
| Stolen Visa card | "card stolen in Lisbon" | `replied` (corpus has contact info), `product_issue` |
| Completely off-topic | "What is the name of the actor in Iron Man?" | `replied`, `invalid`, out-of-scope message |
| Pleasantry | "Thank you for helping me" | `replied`, `invalid` |
| Account security (Google login) | Google login account deletion | `replied` with documented steps |

Key insight: Not all security/account questions require escalation if the corpus documents a safe resolution path. The Visa stolen card case was `replied` because the corpus has the exact contact number. Always check corpus first before deciding to escalate.

---

## 16. Out-of-Scope Handling

When `request_type = invalid`:
- `status` = `replied`
- `response` = "I'm sorry, this question is outside the scope of my support capabilities. [optionally suggest where they might find help]"
- `justification` = "Issue is unrelated to HackerRank, Claude, or Visa support topics"
- `product_area` = `general` or `out_of_scope`

Do NOT escalate purely out-of-scope questions. Reply with the out-of-scope message.

---

## 17. Secrets & Environment

`.env` file (never committed):
```
GEMINI_API_KEY=your-gemini-api-key
```

Always read via:
```python
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
```

Never hardcode API keys.

---

## 18. Submission Checklist

- [ ] `support_tickets/output.csv` has all rows populated with 5 columns
- [ ] `code/README.md` exists with install + run instructions
- [ ] `requirements.txt` with pinned versions
- [ ] No API keys hardcoded anywhere
- [ ] No `data/` or `support_tickets/` CSVs in the code zip
- [ ] No virtualenv or `__pycache__` in code zip
- [ ] `log.txt` at `$HOME/hackerrank_orchestrate/log.txt` exists (chat transcript)
- [ ] Tested against `sample_support_tickets.csv` and outputs match expected
