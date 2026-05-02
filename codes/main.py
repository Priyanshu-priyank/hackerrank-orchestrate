"""
main.py — Entry point for the HackerRank Orchestrate support triage agent.

Usage:
    python main.py              # Run on support_tickets/support_tickets.csv
    python main.py --sample     # Run on sample_support_tickets.csv (shows comparison)
    python main.py --limit 5    # Process only first N rows (for quick testing)

Output:
    support_tickets/output.csv
"""

import os
# Fix #6: Suppress ChromaDB and tokenizer telemetry/noise before any imports
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Fix #5: prevent fork deadlock on Windows/macOS

import argparse
import logging
import sys
import time
from pathlib import Path

# Fix #10: Reconfigure stdout to UTF-8 on Windows terminals (default cp1252 crashes on ✓/✗)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass  # already UTF-8 or reconfigure not supported — safe to ignore

# Fix #12: Silence ChromaDB posthog telemetry at the logger level.
# The env var alone doesn't work when posthog has a version mismatch (capture() signature error).
# Suppressing these loggers stops the "Failed to send telemetry event" noise entirely.
for _noisy_logger in ("chromadb.telemetry", "posthog", "chromadb"):
    logging.getLogger(_noisy_logger).setLevel(logging.CRITICAL)

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add code/ to path so imports work when running from any directory
sys.path.insert(0, str(Path(__file__).parent))

from config import INPUT_CSV, OUTPUT_CSV, SAMPLE_CSV
from retriever import Retriever
from classifier import classify
from agent import run_agent

OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]


def process_tickets(df: pd.DataFrame, retriever: Retriever, sample_mode: bool = False) -> pd.DataFrame:
    """
    Process all rows in df and return a DataFrame with the 5 output columns.
    Prints per-row progress to terminal.
    """
    results = []
    total = len(df)

    print(f"\n{'='*60}")
    print(f"Processing {total} ticket(s)…")
    print(f"{'='*60}\n")

    for idx, row in df.iterrows():
        issue   = str(row.get("issue", "") or "").strip()
        subject = str(row.get("subject", "") or "").strip()
        company = str(row.get("company", "") or "").strip()

        # Pre-LLM classification
        clf = classify(issue, subject, company)

        # Run full pipeline
        t0 = time.time()
        result = run_agent(issue, subject, clf, retriever)
        elapsed = time.time() - t0

        results.append(result)

        # Terminal progress line
        company_display = clf.company or "unknown"
        print(
            f"[{idx+1:>3}/{total}] "
            f"co={company_display:<12} "
            f"status={result['status']:<10} "
            f"type={result['request_type']:<16} "
            f"area={result['product_area']:<20} "
            f"({elapsed:.1f}s)"
        )

        # In sample mode, show expected vs actual for columns that exist
        if sample_mode:
            for col in OUTPUT_COLUMNS:
                expected_col = col  # sample CSV uses same column names
                if expected_col in row and pd.notna(row[expected_col]):
                    expected = str(row[expected_col]).strip().lower()
                    actual   = str(result.get(col, "")).strip().lower()
                    match    = "[PASS]" if expected == actual else "[FAIL]"
                    if col in ("status", "request_type"):  # only flag discrete fields
                        print(f"       {match} {col}: expected={expected!r} got={actual!r}")

    out_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)

    # Preserve original input columns in output (handy for review)
    for col in ["issue", "subject", "company"]:
        if col in df.columns:
            out_df.insert(0, col, df[col].values)

    return out_df


def main():
    parser = argparse.ArgumentParser(description="HackerRank Orchestrate — Support Triage Agent")
    parser.add_argument("--sample", action="store_true", help="Run on sample CSV (shows expected vs actual)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    args = parser.parse_args()

    input_path = SAMPLE_CSV if args.sample else INPUT_CSV

    print(f"\n{'='*60}")
    print("HackerRank Orchestrate — Support Triage Agent")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {OUTPUT_CSV}")

    # Load input CSV
    if not Path(input_path).exists():
        print(f"\n❌ Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path, encoding="utf-8")
    df.columns = df.columns.str.lower().str.strip()  # Fix #1: normalise column headers to lowercase
    print(f"Loaded {len(df)} rows.\n")

    if args.limit:
        df = df.head(args.limit)
        print(f"(Limited to first {args.limit} rows.)\n")

    # Initialise retriever (indexes corpus once, then reuses ChromaDB)
    retriever = Retriever()

    # Run pipeline
    out_df = process_tickets(df, retriever, sample_mode=args.sample)

    # Write output (only the 5 required columns for the real run)
    output_path = OUTPUT_CSV
    if args.sample:
        # In sample mode, write to a separate file to avoid overwriting real output
        output_path = Path(OUTPUT_CSV).parent / "output_sample_test.csv"

    # Write only the required 5 columns to the submission file
    submission_cols = out_df[OUTPUT_COLUMNS] if not args.sample else out_df
    submission_cols.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"✅ Done. Output written to: {output_path}")
    print(f"{'='*60}\n")

    # Summary stats
    status_counts = out_df["status"].value_counts().to_dict()
    type_counts   = out_df["request_type"].value_counts().to_dict()
    print("Summary:")
    print(f"  Status:       {status_counts}")
    print(f"  Request type: {type_counts}\n")


if __name__ == "__main__":
    main()
