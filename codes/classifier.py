"""
classifier.py — Pre-LLM triage signals.

Runs BEFORE the LLM API call to:
  1. Normalise the company field (handle None / aliases)
  2. Detect high-risk keywords that should bias toward escalation
  3. Detect system-wide outage phrases → hard escalate, skip LLM
  4. Detect prompt-injection attempts (treat as invalid)
  5. Infer company from issue text when company == None

Returns a ClassifierResult dataclass consumed by agent.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import (
    COMPANY_ALIASES,
    COMPANY_KEYWORDS,
    HIGH_RISK_KEYWORDS,
    INJECTION_PATTERNS,
    OUTAGE_KEYWORDS,
    VALID_COMPANIES,
)


@dataclass
class ClassifierResult:
    company: str | None            # normalised: "hackerrank" | "claude" | "visa" | None
    is_high_risk: bool             # True → bias toward escalation
    is_injection: bool             # True → treat as invalid, reply out-of-scope
    is_empty: bool                 # True → no usable issue content
    is_outage: bool                # True → hard-escalate, skip LLM
    risk_triggers: list[str] = field(default_factory=list)   # which keywords fired
    inferred_company: bool = False # True if company was guessed from content


def classify(issue: str, subject: str, raw_company: str) -> ClassifierResult:
    """
    Run all pre-LLM checks and return a ClassifierResult.

    Args:
        issue:       raw issue text
        subject:     raw subject text (may be empty)
        raw_company: company field from CSV (may be "None", blank, or a name)
    """
    # ── 1. Normalise company ─────────────────────────────────────────────────
    company, inferred = _normalise_company(issue, subject, raw_company)

    # ── 2. Empty issue check ─────────────────────────────────────────────────
    combined = (issue or "").strip() + " " + (subject or "").strip()
    is_empty = len(combined.strip()) < 5

    # ── 3. Prompt injection detection ────────────────────────────────────────
    is_injection = _detect_injection(issue)

    # ── 4. High-risk keyword detection ───────────────────────────────────────
    is_high_risk, risk_triggers = _detect_high_risk(issue)

    # ── 5. Outage detection ───────────────────────────────────────────────────
    is_outage = _detect_outage(issue, subject)

    return ClassifierResult(
        company=company,
        is_high_risk=is_high_risk,
        is_injection=is_injection,
        is_empty=is_empty,
        is_outage=is_outage,
        risk_triggers=risk_triggers,
        inferred_company=inferred,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_company(issue: str, subject: str, raw: str) -> tuple[str | None, bool]:
    """
    Return (normalised_company, was_inferred).
    If raw is a known alias, use it directly.
    If raw is None/blank, try to infer from issue+subject text.
    """
    raw_lower = (raw or "").strip().lower()

    # Direct alias match
    if raw_lower in COMPANY_ALIASES:
        direct = COMPANY_ALIASES[raw_lower]
        if direct is not None:
            return direct, False
        # raw was None/empty → try to infer
    elif raw_lower in VALID_COMPANIES:
        return raw_lower, False

    # Inference from content
    text = ((issue or "") + " " + (subject or "")).lower()
    best_company, best_score = None, 0

    for company, keywords in COMPANY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_company = company

    return (best_company, True) if best_score > 0 else (None, False)


def _detect_injection(issue: str) -> bool:
    """Return True if the issue contains prompt-injection patterns."""
    text = (issue or "").lower()
    return any(pattern in text for pattern in INJECTION_PATTERNS)


def _detect_high_risk(issue: str) -> tuple[bool, list[str]]:
    """Return (is_high_risk, list_of_matched_keywords)."""
    text = (issue or "").lower()
    triggers = [kw for kw in HIGH_RISK_KEYWORDS if kw in text]
    return (len(triggers) > 0), triggers


def _detect_outage(issue: str, subject: str) -> bool:
    """Return True if the ticket describes a system-wide outage."""
    text = ((issue or "") + " " + (subject or "")).lower()
    return any(phrase in text for phrase in OUTAGE_KEYWORDS)
