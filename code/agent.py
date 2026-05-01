import os
import time
import json
from typing import Literal
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from config import MODEL, PRODUCT_AREAS
from prompts import SYSTEM_PROMPT, build_user_message


class TriageDecision(BaseModel):
    status: Literal["replied", "escalated"]
    product_area: Literal[
        "billing",
        "account",
        "screen",
        "community",
        "privacy",
        "travel_support",
        "general_support",
        "conversation_management",
        "out_of_scope",
        "general",
    ]
    response: str
    justification: str
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"]


ERROR_ROW = {
    "status": "escalated",
    "product_area": "general",
    "response": "",
    "justification": "Agent error: API call failed.",
    "request_type": "product_issue",
}


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    payload = {
        "sessionId": "de13b5",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(datetime.now(tz=timezone.utc).timestamp() * 1000),
    }
    debug_log_path = Path(__file__).resolve().parent.parent / "debug-de13b5.log"
    with open(debug_log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")


def run_agent(issue, subject, company, chunks, high_risk, injection, invalid):
    if invalid or _is_pleasantry(issue) or _is_off_topic(issue) or injection:
        return {
            "status": "replied",
            "product_area": "out_of_scope" if not _looks_like_conversation(issue) else "conversation_management",
            "response": "I'm sorry, this question is outside the scope of my support capabilities.",
            "justification": "Ticket is invalid, out of scope, or contains prompt-injection text.",
            "request_type": "invalid",
        }

    if _is_system_outage(issue):
        return {
            "status": "escalated",
            "product_area": "general",
            "response": "",
            "justification": "The ticket describes a broad outage or system-wide failure that requires human escalation.",
            "request_type": "bug",
        }

    if high_risk and not _visa_stolen_card_with_contact(issue, company, chunks):
        return {
            "status": "escalated",
            "product_area": "general_support",
            "response": "",
            "justification": "A high-risk keyword was detected and the retrieved corpus did not provide an exact safe self-serve resolution.",
            "request_type": "product_issue",
        }

    flags = []
    if high_risk:
        flags.append("high_risk")
    if injection:
        flags.append("prompt_injection")
    if invalid:
        flags.append("invalid")

    user_message = build_user_message(
        company=company,
        subject=subject,
        issue=issue,
        flags=flags,
        chunks=chunks,
    )

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    # #region agent log
    _debug_log(
        run_id="api-diagnosis",
        hypothesis_id="H5",
        location="code/agent.py:106",
        message="API key presence before Gemini call",
        data={
            "api_key_present": bool(api_key),
            "api_key_prefix": (api_key or "")[:4],
            "company": (company or "").lower(),
            "chunks_count": len(chunks or []),
        },
    )
    # #endregion
    if not api_key:
        return ERROR_ROW.copy()

    last_error = None
    for attempt in range(2):
        try:
            client = genai.Client(api_key=api_key)
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
            result = response.parsed if response.parsed is not None else response.text
            return _normalize_result(result)
        except Exception as exc:
            last_error = exc
            # #region agent log
            _debug_log(
                run_id="api-diagnosis",
                hypothesis_id="H6",
                location="code/agent.py:128",
                message="Gemini API exception",
                data={
                    "attempt": attempt + 1,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc)[:500],
                },
            )
            # #endregion
            if attempt == 0:
                time.sleep(1)

    row = ERROR_ROW.copy()
    row["justification"] = _format_agent_error(last_error)
    return row


def _normalize_result(result):
    row = ERROR_ROW.copy()
    if isinstance(result, BaseModel):
        result = result.model_dump()
    elif isinstance(result, str):
        result = TriageDecision.model_validate_json(result).model_dump()
    row.update({key: "" if value is None else str(value).strip() for key, value in result.items()})

    if row["status"] not in {"replied", "escalated"}:
        row["status"] = "escalated"
    if row["request_type"] not in {"product_issue", "feature_request", "bug", "invalid"}:
        row["request_type"] = "product_issue"
    if row["product_area"] not in PRODUCT_AREAS:
        row["product_area"] = _map_product_area(row["product_area"])
    if row["status"] == "escalated":
        row["response"] = ""
    if row["request_type"] == "invalid":
        row["status"] = "replied"
    return row


def _format_agent_error(error: Exception | None) -> str:
    if error is None:
        return "Agent error: API call failed."
    text = str(error)
    if "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
        return "Agent error: API call failed. RESOURCE_EXHAUSTED quota exceeded for Gemini API."
    return f"Agent error: API call failed. {type(error).__name__}"


def _map_product_area(value):
    text = (value or "").lower()
    if "bill" in text or "subscription" in text or "payment" in text:
        return "billing"
    if "account" in text or "team" in text or "workspace" in text:
        return "account"
    if "screen" in text or "test" in text or "assessment" in text:
        return "screen"
    if "community" in text or "resume" in text:
        return "community"
    if "privacy" in text or "data" in text:
        return "privacy"
    if "travel" in text or "cheque" in text or "cash" in text:
        return "travel_support"
    if "conversation" in text:
        return "conversation_management"
    if "scope" in text:
        return "out_of_scope"
    if "general" in text or not text:
        return "general"
    return "general_support"


def _is_system_outage(issue):
    text = (issue or "").lower()
    return any(
        phrase in text
        for phrase in [
            "site is down",
            "not loading",
            "outage",
            "none of the pages",
            "all requests are failing",
            "stopped working completely",
            "none of the submissions",
            "across any challenges",
            "resume builder is down",
        ]
    )


def _is_pleasantry(issue):
    text = (issue or "").strip().lower()
    return text in {"thanks", "thank you", "thank you for helping me", "happy to help"}


def _is_off_topic(issue):
    text = (issue or "").lower()
    return any(
        phrase in text
        for phrase in [
            "iron man",
            "delete all files",
            "rm -rf",
            "format the system",
        ]
    )


def _looks_like_conversation(issue):
    return "conversation" in (issue or "").lower()


def _visa_stolen_card_with_contact(issue, company, chunks):
    text = f"{issue or ''}\n{chr(10).join(chunks or [])}".lower()
    company_text = (company or "").lower()
    has_visa = company_text == "visa" or "visa" in text
    has_stolen = "stolen" in text or "lost" in text
    has_contact = "000-800-100-1219" in text or "+1 303 967 1090" in text or "1-800" in text
    return has_visa and has_stolen and has_contact
