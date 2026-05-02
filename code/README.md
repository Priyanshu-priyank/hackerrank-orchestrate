# Support Triage Agent

This agent uses RAG (Retrieval-Augmented Generation) with ChromaDB, SentenceTransformers, and Google Gemini to automatically triage support tickets for HackerRank, Claude, and Visa.

## Prerequisites
- Python 3.11+
- Gemini API Key

## Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup

Copy `.env.example` to `.env` in the root directory:
```bash
cp ../.env.example ../.env
```
Open `../.env` and set your `GEMINI_API_KEY`.

## Running the Agent

```bash
# Run against the main dataset (writes to support_tickets/output.csv)
python main.py

# Run against the sample dataset for testing
python main.py --sample
```
