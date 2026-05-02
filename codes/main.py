"""
main.py — Entry point for the HackerRank Orchestrate support triage agent.

Usage:
    python main.py              # Run on support_tickets/support_tickets.csv
    python main.py --sample     # Run on sample_support_tickets.csv (shows comparison)
    python main.py --limit 5    # Process only first N rows (for quick testing)

Output:
    support_tickets/output.csv   <- submission file (5 required columns only)
    triage.log                   <- full run log with timestamps (same dir as main.py)

Note on log.txt:
    The hackathon's log.txt at $HOME/hackerrank_orchestrate/log.txt is your AI CHAT
    TRANSCRIPT (conversations with Claude/Cursor). That is written by your AI tool,
    not by this script. triage.log is this agent's runtime log — a separate file.
"""

import os

# Suppress ChromaDB and tokenizer noise BEFORE any other imports
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Fix UTF-8 on Windows terminals (default cp1252 crashes on unicode chars)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Silence ChromaDB posthog telemetry at logger level too
for _noisy in ("chromadb.telemetry", "posthog", "chromadb"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from config import INPUT_CSV, OUTPUT_CSV, SAMPLE_CSV
from retriever import Retriever
from classifier import classify
from agent import run_agent

OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]
REQUIRED_INPUT_COLUMNS = {"issue", "subject", "company"}


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    """
    Configure the 'triage' logger to write to both:
      - stdout  (no timestamp prefix — clean while watching live)
      - triage.log (full timestamped log — saved for review and debugging)
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("triage")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # don't bubble up to root logger

    # File handler: timestamped, captures DEBUG and above
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Console handler: no timestamp prefix, INFO and above only
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── CSV validation ────────────────────────────────────────────────────────────

def validate_csv_columns(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Warn clearly if expected input columns are missing."""
    present = set(df.columns)
    missing = REQUIRED_INPUT_COLUMNS - present
    if missing:
        logger.warning(
            f"CSV is missing expected column(s): {sorted(missing)}. "
            f"Affected rows will be treated as empty. "
            f"Columns found: {sorted(present)}"
        )
    extra = present - REQUIRED_INPUT_COLUMNS - set(OUTPUT_COLUMNS)
    if extra:
        logger.debug(f"CSV has extra columns (ignored): {sorted(extra)}")


# ── Core pipeline ─────────────────────────────────────────────────────────────

def process_tickets(
    df: pd.DataFrame,
    retriever: Retriever,
    logger: logging.Logger,
    sample_mode: bool = False,
) -> pd.DataFrame:
    """
    Process every row through the full RAG + LLM pipeline.
    Logs each result to both console (stdout) and triage.log.
    """
    results = []
    total = len(df)
    run_start = time.time()

    logger.info("")
    logger.info("=" * 64)
    logger.info(f"Processing {total} ticket(s)...")
    logger.info("=" * 64)

    for idx, row in df.iterrows():
        row_num = idx + 1  # 1-based for humans

        issue   = str(row.get("issue",   "") or "").strip()
        subject = str(row.get("subject", "") or "").strip()
        company = str(row.get("company", "") or "").strip()

        clf = classify(issue, subject, company)

        t0 = time.time()
        result = run_agent(issue, subject, clf, retriever)
        elapsed = time.time() - t0

        results.append(result)

        # ── Main progress line ────────────────────────────────────────────
        co_disp  = (clf.company or "unknown").ljust(12)
        inferred = " [inferred]" if clf.inferred_company else ""
        status   = result["status"].ljust(10)
        rtype    = result["request_type"].ljust(16)
        area     = result["product_area"].ljust(20)

        logger.info(
            f"[{row_num:>3}/{total}] "
            f"co={co_disp}{inferred} "
            f"status={status} "
            f"type={rtype} "
            f"area={area} "
            f"({elapsed:.1f}s)"
        )

        # ── Detail lines (log file only — not printed to console) ─────────
        detail_logger = logging.getLogger("triage")

        flags = []
        if clf.is_empty:     flags.append("EMPTY_ISSUE")
        if clf.is_injection: flags.append("PROMPT_INJECTION")
        if clf.is_high_risk: flags.append(f"HIGH_RISK[{', '.join(clf.risk_triggers)}]")
        if flags:
            detail_logger.debug(f"  flags: {' | '.join(flags)}")

        just = result.get("justification", "")
        if just:
            detail_logger.debug(
                f"  justification: {just[:140]}{'...' if len(just) > 140 else ''}"
            )

        if result["status"] == "escalated":
            detail_logger.debug(f"  ESCALATED — no response generated")

        # ── Sample mode: show expected vs actual ──────────────────────────
        if sample_mode:
            for col in ("status", "request_type"):
                if col in row and pd.notna(row[col]):
                    expected = str(row[col]).strip().lower()
                    actual   = str(result.get(col, "")).strip().lower()
                    mark = "PASS" if expected == actual else "FAIL"
                    logger.info(
                        f"       [{mark}] {col}: expected={expected!r}  got={actual!r}"
                    )

    # ── Assemble output DataFrame ─────────────────────────────────────────
    out_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)

    for col in ["issue", "subject", "company"]:
        if col in df.columns:
            out_df.insert(0, col, df[col].values)

    total_elapsed = time.time() - run_start
    logger.info("")
    logger.info(
        f"All {total} tickets processed in {total_elapsed:.1f}s "
        f"(avg {total_elapsed / max(total, 1):.1f}s/ticket)"
    )
    return out_df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HackerRank Orchestrate — Support Triage Agent"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Run on sample_support_tickets.csv and compare expected vs actual",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rows (quick smoke test)",
    )
    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────
    log_file = Path(__file__).parent / "triage.log"
    logger = setup_logging(log_file)

    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("")
    logger.info("=" * 64)
    logger.info("HackerRank Orchestrate — Support Triage Agent")
    logger.info(f"Run started : {run_ts}")
    logger.info(f"Log file    : {log_file}")
    logger.info("=" * 64)

    # ── Paths ─────────────────────────────────────────────────────────────
    input_path  = Path(SAMPLE_CSV if args.sample else INPUT_CSV)
    output_path = (
        Path(OUTPUT_CSV).parent / "output_sample_test.csv"
        if args.sample
        else Path(OUTPUT_CSV)
    )

    logger.info(f"Input  : {input_path}")
    logger.info(f"Output : {output_path}")
    if args.limit:
        logger.info(f"Limit  : first {args.limit} rows (smoke-test mode)")

    # ── Guard: input file must exist ──────────────────────────────────────
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error(
            "Tip: run from the code/ directory, or check the paths in config.py."
        )
        sys.exit(1)

    # ── Guard: output directory must exist (create if missing) ────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv(input_path, encoding="utf-8")
    df.columns = df.columns.str.lower().str.strip()  # normalise headers
    logger.info(f"Loaded {len(df)} rows from {input_path.name}")

    validate_csv_columns(df, logger)

    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Trimmed to first {len(df)} rows.")

    # ── Retriever init ────────────────────────────────────────────────────
    retriever = Retriever()

    # ── Pipeline ──────────────────────────────────────────────────────────
    out_df = process_tickets(df, retriever, logger, sample_mode=args.sample)

    # ── Write output CSV ──────────────────────────────────────────────────
    # Real run  → 5 required columns only (submission format)
    # Sample run → all columns kept for manual review
    if args.sample:
        out_df.to_csv(output_path, index=False)
    else:
        out_df[OUTPUT_COLUMNS].to_csv(output_path, index=False)

    logger.info("")
    logger.info("=" * 64)
    logger.info(f"Done. Output written to: {output_path}")
    logger.info("")

    # ── Summary ───────────────────────────────────────────────────────────
    status_counts = out_df["status"].value_counts().to_dict()
    type_counts   = out_df["request_type"].value_counts().to_dict()
    area_counts   = out_df["product_area"].value_counts().to_dict()

    logger.info("Summary")
    logger.info(f"  Status       : {status_counts}")
    logger.info(f"  Request type : {type_counts}")
    logger.info(f"  Product area : {area_counts}")

    # Escalation details — handy for judge interview prep
    escalated_df = out_df[out_df["status"] == "escalated"]
    if not escalated_df.empty:
        logger.info(f"  Escalated    : {len(escalated_df)} ticket(s) — reasons:")
        for _, esc in escalated_df.iterrows():
            co  = str(esc.get("company", "?"))
            jst = str(esc.get("justification", ""))[:90]
            logger.info(f"    co={co:<12}  {jst}")

    logger.info("=" * 64)
    logger.info(f"Run ended : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Console-only hint about the log file location
    print(f"\n  Full timestamped log saved to: {log_file}\n")


if __name__ == "__main__":
    main()
