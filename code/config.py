import os

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CODE_DIR)

MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 3
MAX_CHUNK_TOKENS = 500
CORPUS_PATH = os.path.join(_ROOT_DIR, "data")
CHROMA_PERSIST_PATH = os.path.join(_CODE_DIR, "chroma_store")
INPUT_CSV = os.path.join(_ROOT_DIR, "support_tickets", "support_tickets.csv")
OUTPUT_CSV = os.path.join(_ROOT_DIR, "support_tickets", "output.csv")
SAMPLE_CSV = os.path.join(_ROOT_DIR, "support_tickets", "sample_support_tickets.csv")

HIGH_RISK_KEYWORDS = [
    "fraud", "unauthorized", "stolen", "hacked", "compromised",
    "lawsuit", "legal action", "billing dispute", "charge back",
    "account deleted", "security breach", "identity theft", "legal"
]
