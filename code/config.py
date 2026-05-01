MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CORPUS_PATH = "../data"
INPUT_CSV = "../support_tickets/support_tickets.csv"
OUTPUT_CSV = "../support_tickets/output.csv"
SAMPLE_CSV = "../support_tickets/sample_support_tickets.csv"
CHROMA_PERSIST_PATH = "./chroma_store"

TOP_K_RETRIEVAL = 3
MAX_CHUNK_TOKENS = 500
MAX_ISSUE_WORDS = 1500

OUTPUT_COLUMNS = [
    "status",
    "product_area",
    "response",
    "justification",
    "request_type",
]

PRODUCT_AREAS = {
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
}
