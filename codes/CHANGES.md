# CHANGES.md — Bug Fixes Applied to Support Triage Agent

> All 9 issues fixed (6 from codes_issues.md + 3 newly discovered).
> Each entry lists: the issue, the file+location, the root cause, and the exact fix applied.

---

## Fix #1 — CSV Column Case Sensitivity
**Severity:** CRITICAL
**File:** `main.py`
**Original issue (codes_issues.md #1):** Yes

### Problem
`pd.read_csv()` preserves the original casing of CSV column headers.
The code accessed columns with lowercase keys: `row.get("issue", "")`, `row.get("subject", "")`,
`row.get("company", "")`. If the CSV ships with capitalized headers (`Issue`, `Subject`, `Company`),
every `.get()` returns `""` and every single ticket hits the empty fast-path:
*"I'm sorry, your message appears to be empty."*

### Fix Applied
```python
# BEFORE
df = pd.read_csv(input_path)

# AFTER
df = pd.read_csv(input_path, encoding="utf-8")
df.columns = df.columns.str.lower().str.strip()  # normalise headers
```
Added `encoding="utf-8"` as a bonus to prevent silent encoding failures on Windows.

---

## Fix #2 — Retry Loop Skipped to Next Key Instead of Retrying Current Key
**Severity:** CRITICAL
**File:** `agent.py` → `_try_gemini_keys()`
**Original issue (codes_issues.md #2):** Yes

### Problem
The rate-limit handler called `time.sleep(2)` then `continue`. Because `continue` advances
a `for` loop, it moved to the **next Gemini key** rather than retrying the same one.
With only one Gemini key, the first 429 immediately exhausted all keys and fell back to
Anthropic (or failed entirely).

### Fix Applied
Replaced the flat `for` loop with a `for key` + inner `for attempt in range(3)` structure.
Rate-limit errors now retry the **same key** up to 3 times with **exponential backoff** (1s, 2s, 4s).
Only non-transient errors (auth failure, bad request, etc.) skip to the next key immediately.

```python
# BEFORE — `continue` jumped to next key every time
for i, key in enumerate(keys, 1):
    try:
        ...
    except Exception as e:
        if "429" in str(e):
            time.sleep(2)
            continue  # ← BUG: moves to next key, not retry

# AFTER — inner loop retries same key
for i, key in enumerate(keys, 1):
    for attempt in range(3):          # retry same key up to 3×
        try:
            ...
        except Exception as e:
            if transient_error(e):
                time.sleep(2 ** attempt)  # exponential: 1s, 2s, 4s
                # inner loop continues → retries same key
            else:
                break  # non-transient → skip to next key
```

---

## Fix #3 — No try/except Around json.loads (Gemini response parsing)
**Severity:** HIGH
**File:** `agent.py` → `_call_gemini()`
**Original issue (codes_issues.md #3):** Yes

### Problem
`parsed = json.loads(raw_text)` had no error handling. If Gemini returned malformed JSON,
extra prose, or an empty response, a `json.JSONDecodeError` propagated up uncaught and
crashed the **entire run** mid-pipeline (not just one ticket).

### Fix Applied
Wrapped in `try/except json.JSONDecodeError`. On failure, logs the raw preview and returns
`None` so the caller gracefully rotates to the next key or falls back to Anthropic.

```python
# BEFORE
parsed = json.loads(raw_text)
return _validate_result(parsed)

# AFTER
try:
    parsed = json.loads(raw_text)
except json.JSONDecodeError as e:
    print(f"[agent] Gemini JSON parse failed: {e} | raw preview: {raw_text[:200]!r}")
    return None  # caller rotates to next key or falls back to Anthropic
return _validate_result(parsed)
```

---

## Fix #4 — 504 Deadline Exceeded Not Treated as Transient Error
**Severity:** MEDIUM
**File:** `agent.py` → `_try_gemini_keys()`
**Original issue (codes_issues.md #4):** Yes

### Problem
The transient-error signal check was:
```python
if any(word in err_str for word in ("quota", "rate", "resource exhausted", "429", "503")):
```
The string `"504"` (HTTP 504 Deadline Exceeded — a common Gemini timeout under load) was
missing. A 504 was treated as a fatal/non-transient error, skipping to the next key with
no sleep instead of retrying.

### Fix Applied
Added `"504"` and `"deadline"` to the signal tuple:
```python
TRANSIENT_SIGNALS = ("quota", "rate", "resource exhausted", "429", "503", "504", "deadline")
```

---

## Fix #5 — sentence-transformers / PyTorch Fork Deadlock on Windows/macOS
**Severity:** ENVIRONMENT
**File:** `main.py` (top-level env vars, set before any imports)
**Original issue (codes_issues.md #5):** Yes

### Problem
On Windows and some macOS setups, `import sentence_transformers` hangs indefinitely
on first run due to HuggingFace's tokenizer using multiprocessing with `fork` semantics
that conflict with PyTorch's thread pool initialisation.

### Fix Applied
Set `TOKENIZERS_PARALLELISM=false` as an OS environment variable at the very top of
`main.py`, before any library imports:
```python
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```
This disables the parallelism that triggers the deadlock. Has no effect on output quality.

---

## Fix #6 — ChromaDB Telemetry Pollutes Terminal Output
**Severity:** LOW
**File:** `main.py` (top-level env vars)
**Original issue (codes_issues.md #6):** Yes

### Problem
ChromaDB prints noisy telemetry warnings and connection error messages to stdout on many
systems (especially in offline/restricted environments), polluting the clean per-ticket
progress output.

### Fix Applied
Two env vars set at the top of `main.py` before `import chromadb`:
```python
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
```

---

## Fix #7 — `MAX_ISSUE_CHARS` NameError Crashes on First Real Ticket
**Severity:** CRITICAL
**File:** `agent.py` (import block)
**Original issue (codes_issues.md):** NOT in original report — newly discovered

### Problem
`agent.py` uses `MAX_ISSUE_CHARS` here:
```python
user_message = build_user_message(
    issue=issue[:MAX_ISSUE_CHARS],
    ...
)
```
But the `from config import ...` block at the top of `agent.py` did **not** include
`MAX_ISSUE_CHARS`. The empty/injection fast-paths in `run_agent()` exit before hitting
this line, which is why it wasn't caught in basic tests. Any real, non-empty, non-injection
ticket would immediately throw:
```
NameError: name 'MAX_ISSUE_CHARS' is not defined
```

### Fix Applied
Added `MAX_ISSUE_CHARS` to the config import:
```python
# BEFORE
from config import (
    GEMINI_MODEL, ANTHROPIC_MODEL, MAX_TOKENS,
    VALID_STATUSES, VALID_REQUEST_TYPES,
    get_gemini_keys, get_anthropic_keys,
)

# AFTER
from config import (
    GEMINI_MODEL, ANTHROPIC_MODEL, MAX_TOKENS, MAX_ISSUE_CHARS,  # ← added
    VALID_STATUSES, VALID_REQUEST_TYPES,
    get_gemini_keys, get_anthropic_keys,
)
```

---

## Fix #8 — ChromaDB `n_results` Crash When Company Has Fewer Than top_k Chunks
**Severity:** HIGH
**File:** `retriever.py` → `query()`
**Original issue (codes_issues.md):** NOT in original report — newly discovered

### Problem
```python
n_results=min(top_k, self.collection.count())
```
`self.collection.count()` returns the **total** number of documents in the collection,
not the number matching the `where` filter. If a company subdirectory only has 2 chunks
indexed (e.g. a small `visa/` corpus) but `TOP_K = 3`, ChromaDB throws:
```
ValueError: n_results 3 cannot be greater than the number of elements
```
This crashed every ticket for that company.

### Fix Applied
Pass `n_results=top_k` directly (without the min cap) when using a `where` filter.
ChromaDB handles the clamping internally when the filter is active.
Wrapped the query in a try/except that catches this and falls back to a global search
(see Fix #9).

---

## Fix #9 — No Fallback When `where` Filter Returns Zero Results
**Severity:** MEDIUM
**File:** `retriever.py` → `query()`
**Original issue (codes_issues.md):** NOT in original report — newly discovered

### Problem
If the company-filtered query returned zero chunks (e.g. company was mis-inferred, or
the specific subdirectory had no matching content), the retriever silently returned `[]`.
The agent then saw "No relevant documentation found" and would often incorrectly escalate
tickets that were fully answerable from the broader corpus.

### Fix Applied
Two-level fallback added to `query()`:

1. **Exception fallback**: if the `where`-filtered query throws (Fix #8 scenario),
   retry immediately without the `where` filter using `min(top_k, total_count)`.

2. **Empty-result fallback**: if the filtered query succeeds but returns zero documents,
   retry without the filter.

```python
# If filtered query returns empty → retry globally
if where and (not results["documents"] or not results["documents"][0]):
    print("[retriever] Company filter returned 0 results — retrying without filter…")
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, self.collection.count()),
        include=["documents", "metadatas", "distances"],
    )
```

---

## Summary Table

| # | Issue | Severity | Source | File Changed |
|---|---|---|---|---|
| 1 | CSV column case sensitivity | Critical | codes_issues.md | `main.py` |
| 2 | Retry loop skipped key instead of retrying | Critical | codes_issues.md | `agent.py` |
| 3 | No try/except around json.loads | High | codes_issues.md | `agent.py` |
| 4 | 504 not in transient error signals | Medium | codes_issues.md | `agent.py` |
| 5 | Sentence-transformers fork deadlock | Environment | codes_issues.md | `main.py` |
| 6 | ChromaDB telemetry noise | Low | codes_issues.md | `main.py` |
| 7 | MAX_ISSUE_CHARS not imported → NameError | **Critical** | Newly found | `agent.py` |
| 8 | n_results crash with where filter | **High** | Newly found | `retriever.py` |
| 9 | No fallback when where filter returns empty | **Medium** | Newly found | `retriever.py` |

### Files Modified
- `main.py` — Fixes #1, #5, #6
- `agent.py` — Fixes #2, #3, #4, #7
- `retriever.py` — Fixes #8, #9
- `CLAUDE.md` — Updated to reflect all fixes and new constraints
- `requirements.md` — Updated edge cases and config sections

### Files NOT Changed
- `classifier.py` — No issues found
- `prompts.py` — No issues found
- `config.py` — No issues found
- `requirements.txt` — No changes needed

---

# v2 Fixes — Applied After First Successful Run

> These 3 issues were found during real test execution after the v1 fixes made the pipeline runnable.

---

## Fix #10 — UnicodeEncodeError Crash on Windows Terminal (--sample mode)
**Severity:** HIGH
**File:** `main.py`
**Source:** codes_issues_v2.md #1

### Problem
`--sample` mode prints `"✓"` and `"✗"` to the terminal for pass/fail comparison.
Windows terminals default to `cp1252` encoding, which cannot encode these Unicode characters.
This caused an immediate crash with:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'
```

### Fix Applied — Two-layer defence:
1. Reconfigure stdout to UTF-8 at startup (works on Python 3.7+):
```python
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
```
2. Replaced `✓`/`✗` with ASCII `[PASS]`/`[FAIL]` — works on every terminal, zero encoding dependency.

---

## Fix #11 — Anthropic API 400 Bad Request (Invalid Model Name)
**Severity:** MEDIUM
**File:** `config.py`
**Source:** codes_issues_v2.md #2

### Problem
`ANTHROPIC_MODEL = "claude-sonnet-4-20250514"` returned HTTP 400 from the live Anthropic API
because this model string does not match any currently deployed model.

### Note on the Gemini / Anthropic fallback situation
If Gemini keys are exhausted (expired, over-quota, or simply not set), the agent falls through
to Anthropic. A 400 error there means the model name was wrong — not necessarily a credits issue.
Both need to be working for the full fallback chain to function.

### Fix Applied
Updated to the current valid Claude Sonnet model string:
```python
# BEFORE
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# AFTER
ANTHROPIC_MODEL = "claude-sonnet-4-6"
```

---

## Fix #12 — ChromaDB Telemetry Warnings Still Printing Despite Env Vars
**Severity:** LOW
**File:** `main.py`
**Source:** codes_issues_v2.md #3

### Problem
Even with `CHROMA_TELEMETRY_DISABLED=1` and `ANONYMIZED_TELEMETRY=False` set, ChromaDB
still printed:
```
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
```
This happens because the underlying `posthog` package has a version mismatch with the
`chromadb` version, and the env var check runs after the broken `capture()` call is already
attempted.

### Fix Applied
Silenced the noisy loggers directly at the Python `logging` level — this is guaranteed to
work regardless of posthog version:
```python
for _noisy_logger in ("chromadb.telemetry", "posthog", "chromadb"):
    logging.getLogger(_noisy_logger).setLevel(logging.CRITICAL)
```

---

## Updated Summary Table (All 12 Fixes)

| # | Issue | Severity | Source | File |
|---|---|---|---|---|
| 1 | CSV column case sensitivity | Critical | v1 report | `main.py` |
| 2 | Retry loop skipped key instead of retrying | Critical | v1 report | `agent.py` |
| 3 | No try/except around json.loads | High | v1 report | `agent.py` |
| 4 | 504 not in transient error signals | Medium | v1 report | `agent.py` |
| 5 | Sentence-transformers fork deadlock | Environment | v1 report | `main.py` |
| 6 | ChromaDB telemetry env var noise | Low | v1 report | `main.py` |
| 7 | MAX_ISSUE_CHARS not imported → NameError | Critical | Newly found v1 | `agent.py` |
| 8 | n_results crash with where filter | High | Newly found v1 | `retriever.py` |
| 9 | No fallback when where filter returns empty | Medium | Newly found v1 | `retriever.py` |
| 10 | UnicodeEncodeError on Windows terminal | High | v2 report | `main.py` |
| 11 | Invalid Anthropic model name → 400 error | Medium | v2 report | `config.py` |
| 12 | ChromaDB telemetry still prints (logger fix) | Low | v2 report | `main.py` |
