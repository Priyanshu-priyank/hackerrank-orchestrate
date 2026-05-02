"""
agent.py — LLM triage call with OpenRouter as the strict provider.

Returns a 5-field structured dict:
    status, product_area, response, justification, request_type

Structured output strategy:
  - OpenRouter: JSON mode (response_format: json_object)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv

from config import (
    OPENROUTER_MODEL,
    MAX_TOKENS,
    MAX_ISSUE_CHARS,  # Fix #7: was missing, caused NameError on first real ticket
    VALID_STATUSES,
    VALID_REQUEST_TYPES,
    get_openrouter_keys,
)
from prompts import SYSTEM_PROMPT, build_user_message
from classifier import ClassifierResult
from retriever import Retriever

load_dotenv()

# ── Output schema (shared description) ───────────────────────────────────────
_SCHEMA_PROPERTIES = {
    "status": {
        "type": "string",
        "enum": ["replied", "escalated"],
        "description": "Whether to reply or escalate to a human agent.",
    },
    "product_area": {
        "type": "string",
        "description": "The most relevant support category.",
    },
    "response": {
        "type": "string",
        "description": "User-facing answer grounded in corpus. Empty string if escalated.",
    },
    "justification": {
        "type": "string",
        "description": "Why this decision was made, traceable to corpus.",
    },
    "request_type": {
        "type": "string",
        "enum": ["product_issue", "feature_request", "bug", "invalid"],
        "description": "Classification of the request.",
    },
}
_REQUIRED_FIELDS = ["status", "product_area", "response", "justification", "request_type"]




# ── Public API ────────────────────────────────────────────────────────────────

def run_agent(
    issue: str,
    subject: str,
    clf: ClassifierResult,
    retriever: Retriever,
) -> dict:
    """
    Full pipeline for one ticket:
      1. Short-circuit for empty/injection cases (no API call)
      2. Retrieve corpus chunks
      3. Try Gemini keys, then Anthropic keys
      4. Validate and return structured result
    """

    # ── Fast path: empty issue ───────────────────────────────────────────────
    if clf.is_empty:
        return _make_result(
            status="replied",
            product_area="general",
            response="I'm sorry, your message appears to be empty. Please describe your issue so I can help you.",
            justification="No issue content was provided in the ticket.",
            request_type="invalid",
        )

    # ── Fast path: prompt injection ──────────────────────────────────────────
    if clf.is_injection:
        return _make_result(
            status="replied",
            product_area="general",
            response="I'm sorry, this question is outside the scope of my support capabilities.",
            justification="Ticket contains prompt injection patterns. Treated as invalid.",
            request_type="invalid",
        )

    # ── Retrieve corpus chunks ───────────────────────────────────────────────
    chunks = retriever.query(text=issue, company=clf.company)
    corpus_text = retriever.format_chunks_for_prompt(chunks)

    # No chunks + high-risk → escalate immediately without LLM call
    if not chunks and clf.is_high_risk:
        return _make_result(
            status="escalated",
            product_area=_guess_product_area(clf.company),
            response="",
            justification=(
                f"No relevant documentation found in corpus. "
                f"High-risk signals detected ({', '.join(clf.risk_triggers)}). "
                "Escalated to human agent."
            ),
            request_type="product_issue",
        )

    # ── Build prompt ─────────────────────────────────────────────────────────
    user_message = build_user_message(
        issue=issue[:MAX_ISSUE_CHARS],
        subject=subject,
        company=clf.company,
        corpus_text=corpus_text,
        risk_triggers=clf.risk_triggers,
        is_injection=clf.is_injection,
    )

    # ── Try providers in order ───────────────────────────────────────────────
    result = _try_openrouter_keys(user_message)
    if result is not None:
        return result

    print("[agent] All OpenRouter keys exhausted — returning safe escalation.")
    return _error_result("All LLM providers failed.")


# ── OpenRouter provider ───────────────────────────────────────────────────────

def _try_openrouter_keys(user_message: str) -> dict | None:
    """Try OpenRouter with same retry logic as Gemini."""
    keys = get_openrouter_keys()
    if not keys:
        print("[agent] No OPENROUTER_API_KEY* found — skipping OpenRouter.")
        return None
    
    TRANSIENT_SIGNALS = ("quota", "rate", "resource exhausted", "429", "503", "504", "deadline")
    
    for i, key in enumerate(keys, 1):
        key_label = f"OpenRouter key {i}/{len(keys)}"
        for attempt in range(3):
            try:
                result = _call_openrouter(key, user_message)
                if result is not None:
                    if i > 1:
                        print(f"[agent] Succeeded with {key_label}.")
                    return result
                break
            except Exception as e:
                err_str = str(e).lower()
                if any(signal in err_str for signal in TRANSIENT_SIGNALS):
                    wait = 2 ** attempt
                    print(f"[agent] {key_label} rate-limit (attempt {attempt+1}/3) — sleeping {wait}s…")
                    time.sleep(wait)
                else:
                    print(f"[agent] {key_label} error: {e} — rotating…")
                    break
    return None

def _call_openrouter(api_key: str, user_message: str) -> dict | None:
    """Call OpenRouter API using OpenAI-compatible format."""
    import requests
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    raw_text = response.json()["choices"][0]["message"]["content"].strip()
    
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"[agent] OpenRouter JSON parse failed: {e}")
        return None
    
    return _validate_result(parsed)



# ── Validation & helpers ──────────────────────────────────────────────────────

def _validate_result(raw: dict) -> dict:
    """Ensure all fields exist and enum values are valid."""
    status = raw.get("status", "escalated")
    if status not in VALID_STATUSES:
        status = "escalated"

    request_type = raw.get("request_type", "product_issue")
    if request_type not in VALID_REQUEST_TYPES:
        request_type = "product_issue"

    response = raw.get("response", "")
    if status == "escalated":
        response = ""

    return _make_result(
        status=status,
        product_area=raw.get("product_area", "general_support"),
        response=response,
        justification=raw.get("justification", ""),
        request_type=request_type,
    )


def _make_result(
    status: str,
    product_area: str,
    response: str,
    justification: str,
    request_type: str,
) -> dict:
    return {
        "status": status,
        "product_area": product_area,
        "response": response,
        "justification": justification,
        "request_type": request_type,
    }


def _error_result(reason: str) -> dict:
    return _make_result(
        status="escalated",
        product_area="general_support",
        response="",
        justification=f"Agent error — escalating for safety. Reason: {reason}",
        request_type="product_issue",
    )


def _guess_product_area(company: str | None) -> str:
    defaults = {
        "hackerrank": "screen",
        "claude": "general",
        "visa": "general_support",
    }
    return defaults.get(company or "", "general_support")
