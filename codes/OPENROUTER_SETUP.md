# OpenRouter Integration Summary

This document summarizes the changes made to transition the support triage agent from direct SDKs (Gemini/Anthropic) to the **OpenRouter API**.

## 1. Environment Configuration (`.env`)
- Added `OPENROUTER_API_KEY` to store the API key.
- The system supports multiple keys via `OPENROUTER_API_KEY_1`, `OPENROUTER_API_KEY_2`, etc., for automatic rotation.

## 2. Configuration (`codes/config.py`)
- **[NEW]** Added `OPENROUTER_MODEL`: Defined as `"google/gemini-2.0-flash-001"` (or any other OpenRouter-supported model string).
- **[NEW]** Added `get_openrouter_keys()`: A helper function that collects all available OpenRouter keys from environment variables for failover logic.
- Updated documentation headers to reflect the OpenRouter-first strategy.

## 3. Agent Logic (`codes/agent.py`)
- **[REMOVED]** All direct dependencies on `google-generativeai` and the `anthropic` SDK.
- **[ADDED]** `_call_openrouter(api_key, user_message)`:
  - Uses the `requests` library to send POST requests to `https://openrouter.ai/api/v1/chat/completions`.
  - Implements **JSON Mode** (`response_format: {"type": "json_object"}`) to ensure structured output.
  - Sets `temperature: 0` for deterministic results.
- **[ADDED]** `_try_openrouter_keys(user_message)`:
  - Implements a robust rotation and retry mechanism.
  - Specifically handles transient errors (429 Rate Limit, 503/504 Timeouts) with exponential backoff.
- **[UPDATED]** `run_agent`: Modified to call the OpenRouter pipeline exclusively.

## 4. Dependencies (`codes/requirements.txt`)
- Verified `requests` is included for API calls.
- Direct LLM SDKs (like `google-generativeai`) are no longer strictly required for the core triage logic.

## 5. Verification Tools
- **`codes/list_models.py`**: A small script to list available models from OpenRouter to verify key connectivity.
- **`codes/debug_openrouter.py`**: A dedicated script to test the OpenRouter connection and JSON parsing in isolation.

---
*Last updated: 2026-05-02*
