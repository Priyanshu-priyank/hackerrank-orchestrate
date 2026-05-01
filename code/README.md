# Support Triage Agent

Terminal-based Python support triage agent for HackerRank Orchestrate.

## Prerequisites

- Python 3.11+
- Gemini API key

## Install

```bash
pip install -r requirements.txt
```

## Environment

From the repository root:

```bash
cp .env.example .env
```

Add:

```text
GEMINI_API_KEY=your-gemini-api-key
```

## Run

```bash
cd code && python main.py
```

Output:

```text
../support_tickets/output.csv
```

## Test

```bash
cd code && python main.py --sample
```
