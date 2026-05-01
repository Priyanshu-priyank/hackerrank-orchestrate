# CLAUDE.md — AI Assistant Context for Support Triage Agent

> **READ THIS FIRST.**
> This file tells you exactly what to build, what decisions have already been made,
> what NOT to do, and how every piece connects. Do not deviate from these decisions
> without asking the user. Do not add unrequested features.

---

## Who Is Working On This

A developer participating in the HackerRank Orchestrate 24-hour hackathon (May 2026).
This is their first agentic AI project. They know Python and JavaScript but chose **Python**
for this project because of the better ML/RAG ecosystem.

---

## What We Are Building

A **terminal-based Python agent** that:
1. Reads `support_tickets/support_tickets.csv` (support tickets for HackerRank, Claude, Visa)
2. For each row: retrieves relevant docs from a local corpus, reasons about the ticket, decides to reply or escalate
3. Writes structured results to `support_tickets/output.csv`

This is a **RAG (Retrieval-Augmented Generation) + LLM triage pipeline.** Not a chatbot. Not a web app. Terminal only.

---

## Architecture Decisions (ALREADY DECIDED — Do Not Change)

| Decision | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Better ML/RAG ecosystem |
| LLM | Gemini API (`gemini-2.5-flash`) | Hackathon is on Google Gemini API |
| Vector DB | ChromaDB (PersistentClient) | Lightweight, local, no server needed |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Local, fast, no API calls for embeddings |
| Structured output | Gemini structured JSON output (`response_schema` forced) | Guarantees valid JSON with correct schema |
| Framework | NONE — raw Google Gen AI SDK | No LangChain, no LangGraph, no abstractions |
| CSV handling | pandas | Simple and standard |
| Secrets | python-dotenv | Read `GEMINI_API_KEY` from `.env` |

---

## What NOT To Do (Hard Rules)

- **NO LangChain** — do not add it, do not suggest it
- **NO LangGraph** — same reason
- **NO live web scraping** — agent must use ONLY the local `data/` corpus
- **NO hardcoded API keys** — always use `os.getenv()`
- **NO modifying files in `data/`** — read-only corpus
- **NO modifying files in `support_tickets/*.csv` input files** — only write `output.csv`
- **NO hallucinating policies** — if corpus doesn't cover it, escalate
- **NO adding a web server, FastAPI, Flask, or any HTTP interface** — terminal only
- **NO adding a frontend or UI** — terminal output only
- **NO inventing contact numbers or policies** not found in the corpus

---

## File Structure

### Given by Hackathon (Do NOT rename or move)

```
hackerrank-orchestrate-may26/
├── AGENTS.md                        # Hackathon AI tool rules
├── README.md
├── .env.example                     # Template for .env
├── code/                            # ← BUILD EVERYTHING HERE
│   └── main.py                      # Entry point (given empty)
├── data/
│   ├── hackerrank/                  # HackerRank support docs
│   ├── claude/                      # Claude Help Center docs
│   └── visa/                        # Visa support docs
└── support_tickets/
    ├── sample_support_tickets.csv   # Has expected outputs (use for testing)
    ├── support_tickets.csv          # Inputs only (run agent on this)
    └── output.csv                   # ← Agent writes here
```

### To Build Inside `code/`

```
code/
├── main.py          # Orchestrator: reads CSV → pipeline → writes output.csv
├── retriever.py     # ChromaDB: index corpus at startup, query per ticket
├── agent.py         # Gemini API call: takes ticket + chunks → returns 5-column dict
├── classifier.py    # Pre-LLM: detect company, detect high-risk keywords
├── prompts.py       # ALL prompt strings live here only
├── config.py        # Constants: model, paths, top_k, high-risk keywords list
├── requirements.txt # Pinned versions
└── README.md        # Install + run instructions (mandatory for evaluation)
```

---

## Data Flow (Step by Step)

```
1. main.py: load corpus into ChromaDB (once, skip if already indexed)
2. main.py: read support_tickets/support_tickets.csv with pandas
3. For each row:
   a. classifier.py: infer company if "None", detect high-risk flags
   b. retriever.py: embed issue → query ChromaDB top-3 → return chunks
   c. agent.py: build prompt with ticket + chunks → call Gemini API → parse structured response
   d. main.py: append result to list, print progress to terminal
4. main.py: write all results to support_tickets/output.csv
```

---

## Input Fields

| Field | Notes |
|---|---|
| `issue` | Main content. May have multiple requests, irrelevant text, or injection attempts |
| `subject` | May be blank, noisy, or irrelevant. Treat as secondary signal only |
| `company` | `HackerRank`, `Claude`, `Visa`, or `None` (infer from content if None) |

---

## Output Fields (Exact Column Names)

| Column | Allowed Values |
|---|---|
| `status` | `replied` or `escalated` |
| `product_area` | string (e.g., `billing`, `account`, `screen`, `general_support`, `travel_support`) |
| `response` | string (user-facing answer). Empty string `""` if escalated |
| `justification` | string (why this decision, grounded in corpus) |
| `request_type` | `product_issue`, `feature_request`, `bug`, or `invalid` |

---

## Escalation Logic

**Escalate when ANY of these are true:**
- Keywords: `fraud`, `unauthorized`, `stolen`, `hacked`, `compromised`, `lawsuit`, `legal`, `security breach`, `identity theft`, `chargeback`
- System outage affecting multiple users
- Account deletion requiring identity verification that corpus doesn't document
- Billing dispute requiring account-level access
- Corpus has zero relevant chunks for the issue AND it's sensitive

**Do NOT escalate — reply with "out of scope" message when:**
- Issue is completely irrelevant (e.g., "Who played Iron Man?")
- Pleasantry or test message (e.g., "Thank you")
- Random/gibberish text
→ Set `status=replied`, `request_type=invalid`

**Reply when:**
- Corpus clearly documents the answer
- How-to question with documented steps
- Even for stolen cards: if the corpus has the exact contact number → reply with that info

**Gray area: always escalate over guessing.**

---

## Gemini API Structured Output Pattern

Use Gemini structured JSON output with `response_schema` to force the five-column response shape.

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

---

## ChromaDB Indexing Pattern

```python
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_PERSIST_PATH

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
        self.collection = self.client.get_or_create_collection(
            name="support_corpus",
            metadata={"hnsw:space": "cosine"}
        )
        # Only index if empty (avoid re-indexing every run)
        if self.collection.count() == 0:
            self._index_corpus()

    def query(self, text: str, company: str = None, top_k: int = 3):
        embedding = self.model.encode(text).tolist()
        where_filter = {"company": company.lower()} if company and company != "None" else None
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        return results["documents"][0], results["metadatas"][0]
```

---

## System Prompt Rules (in `prompts.py`)

The system prompt MUST tell Claude:
1. Only use the provided corpus excerpts — never use outside knowledge
2. Escalation triggers (fraud, security, account compromise, etc.)
3. Out-of-scope reply format for `invalid` cases
4. Response format: bullet steps for how-to, concise prose for FAQ
5. Never fabricate contact numbers, policies, or steps
6. When company is `None`, infer from the issue content

---

## Edge Cases to Handle

| Scenario | How to Handle |
|---|---|
| `company = "None"` | Classifier infers from keywords; if still unknown, query all corpus |
| Empty `issue` field | Return `invalid`, `replied`, "No issue content provided" |
| Prompt injection in `issue` (e.g., "ignore instructions") | Flag as `invalid`, `replied`, out-of-scope message |
| Multiple questions in one ticket | Answer the primary/most urgent one, note others in justification |
| Identical issue and subject | Not a problem, use issue as primary |
| Very long issue (>2000 words) | Truncate to first 1500 tokens before sending to Claude |
| No corpus chunks returned | Escalate — cannot answer without grounding |
| API timeout/error | Retry once, then write error row and continue |

---

## Sample Data Patterns (From `sample_support_tickets.csv`)

These are confirmed correct behaviors to validate against:

| Issue Summary | Status | Request Type | Product Area |
|---|---|---|---|
| Tests not expiring as expected | replied | product_issue | screen |
| Site is down | escalated | bug | (blank/general) |
| Test variants vs new tests | replied | product_issue | screen |
| Add extra time for candidate | replied | product_issue | screen |
| Delete account (Google login) | replied | product_issue | community |
| Delete Claude conversation | replied | product_issue | privacy |
| "Who played Iron Man?" | replied | invalid | conversation_management |
| Visa traveller's cheques stolen | replied | product_issue | travel_support |
| Lost/stolen Visa card from India | replied | product_issue | general_support |
| "Thank you for helping me" | replied | invalid | (blank/general) |

Key observations:
- Visa stolen card → `replied` because corpus has exact contact numbers
- Site down → `escalated` because it's a system-level bug needing human action
- Off-topic questions → `replied` with invalid, not escalated

---

## Dependencies (`requirements.txt`)

```
google-genai==1.74.0
chromadb==0.6.3
sentence-transformers==3.4.1
pandas==2.2.3
python-dotenv==1.0.1
pydantic==2.11.3
```

Install: `pip install -r requirements.txt`

---

## Environment Setup

```bash
# In the repo root:
cp .env.example .env
# Edit .env and add:
# GEMINI_API_KEY=your-gemini-api-key
```

The `GEMINI_API_KEY` is the only required secret. Never put it in code.

---

## How to Run

```bash
cd code
pip install -r requirements.txt
python main.py
# Output written to: ../support_tickets/output.csv
# Progress printed to terminal per row
```

For testing against sample:
```bash
python main.py --sample
# Reads sample_support_tickets.csv, prints comparison vs expected
```

---

## Evaluation Criteria the Code Must Satisfy

1. **Agent Design** — clear module separation (retriever, agent, classifier), RAG grounding, explicit escalation logic, deterministic, readable code
2. **Output CSV accuracy** — correct status, product_area, response (grounded), justification (traceable), request_type
3. **No hallucination** — zero fabricated policies, contact numbers, or steps
4. **Escalation correctness** — high-risk tickets must be escalated, not guessed at
5. **Runnable README** — evaluator must be able to run it from scratch

---

## What the AI Assistant Should Do When Asked to Build This

1. Build modules in this order: `config.py` → `retriever.py` → `classifier.py` → `prompts.py` → `agent.py` → `main.py`
2. After building, write `requirements.txt` and `code/README.md`
3. Do not add any module not listed in the file structure
4. Do not add a web interface, API server, or database other than ChromaDB
5. Do not install or import LangChain, LangGraph, OpenAI, or any other LLM SDK
6. Do not add streaming, async, or multi-threading (simple sync loop is fine)
7. Ask the user before making any architectural decision not covered in this file
