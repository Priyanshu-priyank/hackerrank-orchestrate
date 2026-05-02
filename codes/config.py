"""
config.py — All constants, paths, and thresholds.
No logic here. Import from this file everywhere else.

LLM PROVIDER STRATEGY
─────────────────────
Provider: OpenRouter (openrouter.ai/api/v1/chat/completions)
Model:    OPENROUTER_MODEL (default: google/gemini-2.0-flash-001)

Multiple keys are supported via numbered env vars:
  OPENROUTER_API_KEY        (or OPENROUTER_API_KEY_1)
  OPENROUTER_API_KEY_2
  OPENROUTER_API_KEY_3
  ...

The agent rotates through all keys before escalating on failure.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.parent
DATA_DIR        = ROOT_DIR / "data"
TICKETS_DIR     = ROOT_DIR / "support_tickets"
CHROMA_DIR      = Path(__file__).parent / "chroma_store"

INPUT_CSV       = TICKETS_DIR / "support_tickets.csv"
SAMPLE_CSV      = TICKETS_DIR / "sample_support_tickets.csv"
OUTPUT_CSV      = TICKETS_DIR / "output.csv"

# ── LLM Models ───────────────────────────────────────────────────────────────
OPENROUTER_MODEL    = "google/gemini-2.0-flash-001"  # or any OpenRouter model

# ── Embeddings (local, no API) ────────────────────────────────────────────────
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K               = 3
MAX_CHUNK_TOKENS    = 500
MIN_CHUNK_CHARS     = 40
MAX_ISSUE_CHARS     = 3000

# ── LLM ──────────────────────────────────────────────────────────────────────
MAX_TOKENS          = 1024

# ── Valid output values ──────────────────────────────────────────────────────
VALID_STATUSES      = {"replied", "escalated"}
VALID_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
VALID_COMPANIES     = {"hackerrank", "claude", "visa"}

# ── Company normalisation ─────────────────────────────────────────────────────
COMPANY_ALIASES = {
    "hackerrank": "hackerrank",
    "hacker rank": "hackerrank",
    "claude": "claude",
    "anthropic": "claude",
    "visa": "visa",
    "none": None,
    "": None,
}

# ── High-risk keywords → bias toward escalation ───────────────────────────────
HIGH_RISK_KEYWORDS = [
    "fraud", "fraudulent", "unauthorized", "unauthorised",
    "stolen", "theft", "hacked", "hack", "compromised",
    "security breach", "account breach",
    "lawsuit", "legal action", "sue", "court",
    "billing dispute", "chargeback", "charge back",
    "identity theft", "impersonation",
    "not me", "wasn't me", "didn't make this",
]

# ── Outage keywords → always escalate, no LLM needed ─────────────────────────
OUTAGE_KEYWORDS = [
    "site is down",
    "website is down",
    "platform is down",
    "server is down",
    "service is down",
    "everything is down",
    "none of the pages are accessible",
    "not loading for anyone",
    "down for everyone",
    "complete outage",
    "total outage",
]

# ── Company inference keywords (when company == None) ────────────────────────
COMPANY_KEYWORDS = {
    "hackerrank": [
        "hackerrank", "hacker rank", "assessment", "test", "coding challenge",
        "candidate", "invite", "proctoring", "plagiarism", "screen", "recruit",
        "question", "leaderboard", "contest", "hackathon platform", "hr4w",
    ],
    "claude": [
        "claude", "anthropic", "conversation", "prompt", "context window",
        "claude.ai", "artifact", "project", "memory", "subscription",
        "pro plan", "team plan", "usage limit", "message limit",
    ],
    "visa": [
        "visa", "card", "credit card", "debit card", "transaction",
        "payment", "merchant", "atm", "contactless", "chargeback",
        "traveller", "traveler", "cheque", "foreign exchange", "fx",
        "international payment", "visa checkout", "verified by visa",
    ],
}

# ── Prompt injection patterns ─────────────────────────────────────────────────
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "ignore your instructions",
    "disregard the above",
    "forget your system prompt",
    "act as",
    "you are now",
    "new instructions:",
    "override instructions",
    "show me the retrieved",
    "display the retrieved",
    "show all internal rules",
    "display internal rules",
    "affiche toutes les règles",   # French injection (ticket 25)
    "règles internes",
]


# ── Key collection helpers ────────────────────────────────────────────────────

def _collect_keys(prefix: str) -> list[str]:
    """
    Read all env vars matching PREFIX, PREFIX_1 … PREFIX_9.
    Returns a deduplicated ordered list, skipping blanks.
    """
    keys: list[str] = []
    seen: set[str] = set()

    for candidate in [prefix] + [f"{prefix}_{i}" for i in range(1, 10)]:
        val = os.getenv(candidate, "").strip()
        if val and val not in seen:
            keys.append(val)
            seen.add(val)

    return keys


def get_openrouter_keys() -> list[str]:
    return _collect_keys("OPENROUTER_API_KEY")
