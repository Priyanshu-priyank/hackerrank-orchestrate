"""
agent.py — LLM triage call with OpenRouter as the strict provider.

Returns a 5-field structured dict:
    status, product_area, response, justification, request_type

Structured output strategy:
  - OpenRouter: JSON mode (response_format: json_object)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

from config import (
    OPENROUTER_MODEL,
    MAX_TOKENS,
    MAX_ISSUE_CHARS,
    VALID_STATUSES,
    VALID_REQUEST_TYPES,
    get_openrouter_keys,
)
from prompts import SYSTEM_PROMPT, build_user_message
from classifier import ClassifierResult
from retriever import Retriever

load_dotenv()

logger = logging.getLogger("triage.agent")

# ── Required output fields ────────────────────────────────────────────────────
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
      1. Short-circuit for empty/injection/outage cases (no API call)
      2. Retrieve corpus chunks
      3. Try OpenRouter keys with rotation and retry
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
            justification="Ticket contains prompt injection or instruction-override patterns. Treated as invalid per security policy.",
            request_type="invalid",
        )

    # ── Fast path: system-wide outage ────────────────────────────────────────
    if clf.is_outage:
        return _make_result(
            status="escalated",
            product_area="general",
            response="",
            justification=(
                "Ticket describes a system-wide outage or complete service unavailability. "
                "This requires immediate human intervention and cannot be resolved through automated support."
            ),
            request_type="bug",
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
                "Escalated to human agent for safety."
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

    # ── Try OpenRouter ───────────────────────────────────────────────────────
    outcome = _try_openrouter_keys(user_message)

    if isinstance(outcome, dict):
        return outcome

    # outcome is either None (no keys configured) or a str (last error reason)
    reason = outcome if isinstance(outcome, str) else "No OPENROUTER_API_KEY found in .env — add your key and restart."
    logger.error(f"[agent] All OpenRouter keys exhausted. Reason: {reason}")
    return _error_result(reason)


# ── OpenRouter provider ───────────────────────────────────────────────────────

def _try_openrouter_keys(user_message: str) -> dict | None:
    """Try OpenRouter keys with retry and rotation logic."""
    keys = get_openrouter_keys()
    if not keys:
        logger.warning("[agent] No OPENROUTER_API_KEY* found — skipping OpenRouter.")
        return None

    # Transient errors worth retrying with backoff
    TRANSIENT_SIGNALS = (
        "quota", "rate", "resource exhausted",
        "429", "503", "504", "deadline", "timeout",
    )
    # Permanent errors — rotating keys won't help
    PERMANENT_SIGNALS = {
        "401": "Invalid API key — check OPENROUTER_API_KEY in your .env",
        "402": "No OpenRouter credits — top up at openrouter.ai/credits",
        "400": "Bad request — model may not support json_object mode; try a different OPENROUTER_MODEL",
    }

    last_error_reason = "Unknown error"

    for i, key in enumerate(keys, 1):
        key_label = f"OpenRouter key {i}/{len(keys)}"
        for attempt in range(3):
            try:
                result = _call_openrouter(key, user_message)
                if result is not None:
                    if i > 1:
                        logger.info(f"[agent] Succeeded with {key_label}.")
                    return result
                break
            except Exception as e:
                err_str = str(e).lower()

                # Check for permanent errors first — no point retrying
                for code, human_reason in PERMANENT_SIGNALS.items():
                    if code in err_str:
                        logger.error(f"[agent] {key_label} permanent error ({code}): {human_reason}")
                        last_error_reason = human_reason
                        break
                else:
                    # Not a permanent error — check if transient
                    if any(signal in err_str for signal in TRANSIENT_SIGNALS):
                        wait = 2 ** attempt
                        logger.warning(
                            f"[agent] {key_label} transient error (attempt {attempt+1}/3) — "
                            f"sleeping {wait}s… ({e})"
                        )
                        time.sleep(wait)
                        last_error_reason = str(e)
                        continue
                    else:
                        logger.error(f"[agent] {key_label} error: {e} — rotating key…")
                        last_error_reason = str(e)
                break  # permanent error or exhausted retries — try next key

    return last_error_reason


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
            {"role": "user", "content": user_message},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    raw_text = response.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if the model wraps JSON in ```json ... ```
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        raw_text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.warning(f"[agent] OpenRouter JSON parse failed: {e} — raw: {raw_text[:200]}")
        return None

    return _validate_result(parsed)


# ── Validation & helpers ──────────────────────────────────────────────────────

def _validate_result(raw: dict) -> dict:
    """Ensure all fields exist, enum values are valid, and justification is populated."""
    status = raw.get("status", "escalated")
    if status not in VALID_STATUSES:
        status = "escalated"

    request_type = raw.get("request_type", "product_issue")
    if request_type not in VALID_REQUEST_TYPES:
        request_type = "product_issue"

    response_text = raw.get("response", "")
    if status == "escalated":
        response_text = ""

    # Justification: try multiple field names the LLM might use
    justification = (
        raw.get("justification")
        or raw.get("reasoning")
        or raw.get("explanation")
        or raw.get("reason")
        or ""
    )

    return _make_result(
        status=str(status or "escalated"),
        product_area=str(raw.get("product_area") or "general_support"),
        response=str(response_text or ""),
        justification=str(justification or ""),
        request_type=str(request_type or "product_issue"),
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
    """
    Safe escalation when LLM pipeline fails entirely.
    CSV justification is kept clean for evaluators; technical reason is logged by the caller.
    """
    return _make_result(
        status="escalated",
        product_area="general_support",
        response="",
        justification=(
            "Unable to process this ticket automatically. "
            "Escalated to a human agent for review."
        ),
        request_type="product_issue",
    )


def _guess_product_area(company: str | None) -> str:
    defaults = {
        "hackerrank": "screen",
        "claude": "general",
        "visa": "general_support",
    }
    return defaults.get(company or "", "general_support")
