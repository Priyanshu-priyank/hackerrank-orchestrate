import sys
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from agent import run_agent
from classifier import Classifier
from config import (
    CORPUS_PATH,
    INPUT_CSV,
    MAX_ISSUE_WORDS,
    OUTPUT_COLUMNS,
    OUTPUT_CSV,
    SAMPLE_CSV,
)
from retriever import Retriever


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


def _resolve_from_code_dir(path_value: str) -> str:
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((Path(__file__).resolve().parent / path_obj).resolve())


def main(sample_mode=False):
    # #region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H1",
        location="code/main.py:36",
        message="Environment state before load_dotenv",
        data={
            "python_terminal_use_env_file": "unknown_from_runtime",
            "gemini_key_present_before": bool(os.getenv("GEMINI_API_KEY")),
        },
    )
    # #endregion
    load_dotenv()
    # #region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H2",
        location="code/main.py:49",
        message="Environment state after load_dotenv",
        data={
            "gemini_key_present_after": bool(os.getenv("GEMINI_API_KEY")),
            "gemini_key_prefix": (os.getenv("GEMINI_API_KEY") or "")[:4],
            "env_file_exists": os.path.exists("../.env"),
        },
    )
    # #endregion
    resolved_corpus_path = _resolve_from_code_dir(CORPUS_PATH)
    resolved_input_csv = _resolve_from_code_dir(SAMPLE_CSV if sample_mode else INPUT_CSV)
    resolved_output_csv = _resolve_from_code_dir(OUTPUT_CSV)
    # #region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H4",
        location="code/main.py:73",
        message="Resolved file paths from code directory",
        data={
            "resolved_corpus_path": resolved_corpus_path,
            "resolved_input_csv": resolved_input_csv,
            "resolved_output_csv": resolved_output_csv,
            "resolved_input_exists": os.path.exists(resolved_input_csv),
        },
    )
    # #endregion
    retriever = Retriever(resolved_corpus_path)
    classifier = Classifier()
    input_file = resolved_input_csv
    # #region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H3",
        location="code/main.py:66",
        message="CSV path diagnostics before read_csv",
        data={
            "cwd": os.getcwd(),
            "input_file_raw": input_file,
            "input_file_abs": os.path.abspath(input_file),
            "input_file_exists": os.path.exists(input_file),
            "input_file_abs_exists": os.path.exists(os.path.abspath(input_file)),
        },
    )
    # #endregion
    df = pd.read_csv(input_file, encoding="utf-8").fillna("")
    df.columns = [column.strip().lower() for column in df.columns]

    results = []
    for i, row in df.iterrows():
        issue = str(row.get("issue", "")).strip()
        subject = str(row.get("subject", "")).strip()
        company = str(row.get("company", "None")).strip()

        words = issue.split()
        if len(words) > MAX_ISSUE_WORDS:
            issue = " ".join(words[:MAX_ISSUE_WORDS])

        if company == "None" or not company:
            company = classifier.infer_company(issue, subject) or "unknown"

        high_risk = classifier.is_high_risk(issue)
        injection = classifier.is_prompt_injection(issue)
        invalid = classifier.is_invalid(issue)

        chunks, metadatas = retriever.query(issue, company=company)
        result = run_agent(issue, subject, company, chunks, high_risk, injection, invalid)
        result = {column: result.get(column, "") for column in OUTPUT_COLUMNS}
        results.append(result)

        print(f"[{i}] {result['status']} | {result['request_type']} | {result['product_area']}")

    pd.DataFrame(results, columns=OUTPUT_COLUMNS).to_csv(
        resolved_output_csv,
        index=False,
        encoding="utf-8",
    )
    print(f"\nDone. Output: {resolved_output_csv}")


if __name__ == "__main__":
    sample_mode = "--sample" in sys.argv
    main(sample_mode=sample_mode)

