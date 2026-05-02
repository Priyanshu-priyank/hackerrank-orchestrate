"""
retriever.py — Corpus indexing and retrieval using ChromaDB + sentence-transformers.

Responsibilities:
  - Walk data/ directory and chunk all documents at startup
  - Embed chunks with all-MiniLM-L6-v2 (local, no API calls)
  - Store vectors + metadata in a persistent ChromaDB collection
  - Query by issue text, with optional company filter
"""

import os
import re
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    DATA_DIR,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_CHARS,
    VALID_COMPANIES,
)


class Retriever:
    COLLECTION_NAME = "support_corpus"

    def __init__(self):
        print("[retriever] Loading embedding model…")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"[retriever] Opening ChromaDB at {CHROMA_DIR}")
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        if self.collection.count() == 0:
            print("[retriever] Corpus not indexed yet — indexing now…")
            self._index_corpus()
            print(f"[retriever] Indexed {self.collection.count()} chunks.")
        else:
            print(f"[retriever] Using existing index ({self.collection.count()} chunks).")

    # ─────────────────────────────────────────────────────────────────────────
    # Indexing
    # ─────────────────────────────────────────────────────────────────────────

    def _index_corpus(self):
        """Walk data/{hackerrank,claude,visa}/ and index every text file."""
        ids, docs, embeddings, metadatas = [], [], [], []
        chunk_id = 0

        for company_dir in sorted(DATA_DIR.iterdir()):
            if not company_dir.is_dir():
                continue
            company = company_dir.name.lower()
            if company not in VALID_COMPANIES:
                continue

            for filepath in self._walk_text_files(company_dir):
                text = self._read_file(filepath)
                if not text:
                    continue

                chunks = self._chunk_text(text)
                embeddings_batch = self.model.encode(chunks, show_progress_bar=False)

                for i, (chunk, emb) in enumerate(zip(chunks, embeddings_batch)):
                    ids.append(f"chunk_{chunk_id}")
                    docs.append(chunk)
                    embeddings.append(emb.tolist())
                    metadatas.append({
                        "company": company,
                        "filename": filepath.name,
                        "product_area": self._infer_product_area(filepath.name, chunk),
                        "chunk_index": i,
                    })
                    chunk_id += 1

                    # Batch upsert every 500 chunks to stay memory-friendly
                    if len(ids) >= 500:
                        self.collection.add(
                            ids=ids,
                            documents=docs,
                            embeddings=embeddings,
                            metadatas=metadatas,
                        )
                        ids, docs, embeddings, metadatas = [], [], [], []

        # Flush remainder
        if ids:
            self.collection.add(
                ids=ids,
                documents=docs,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    def _walk_text_files(self, base_dir: Path):
        """Recursively yield all text/markdown/html files."""
        EXTENSIONS = {".txt", ".md", ".html", ".htm", ".rst", ".csv"}
        for root, _, files in os.walk(base_dir):
            for fname in sorted(files):
                if Path(fname).suffix.lower() in EXTENSIONS:
                    yield Path(root) / fname

    def _read_file(self, filepath: Path) -> str:
        """Read file, strip HTML tags if needed, normalise whitespace."""
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"[retriever] Warning: could not read {filepath}: {e}")
            return ""

        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks ≤ MAX_CHUNK_TOKENS words.
        Strategy: split on double newlines (paragraphs), then merge small
        adjacent paragraphs, and split oversized ones on sentence boundaries.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            combined_len = len(current.split()) + len(para.split())
            if combined_len <= MAX_CHUNK_TOKENS:
                current = (current + "\n\n" + para).strip() if current else para
            else:
                if current:
                    chunks.append(current)
                # If this paragraph alone is too big, split on sentences
                if len(para.split()) > MAX_CHUNK_TOKENS:
                    chunks.extend(self._split_long_paragraph(para))
                    current = ""
                else:
                    current = para

        if current:
            chunks.append(current)

        return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]

    def _split_long_paragraph(self, text: str) -> list[str]:
        """Split an oversized paragraph on sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""
        for sent in sentences:
            if len((current + " " + sent).split()) > MAX_CHUNK_TOKENS and current:
                chunks.append(current.strip())
                current = sent
            else:
                current = (current + " " + sent).strip()
        if current:
            chunks.append(current.strip())
        return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]

    def _infer_product_area(self, filename: str, chunk_text: str) -> str:
        """Guess product area from filename and chunk content."""
        name = filename.lower()
        text = chunk_text.lower()

        AREA_SIGNALS = {
            "billing": ["billing", "invoice", "payment", "charge", "subscription", "refund", "price"],
            "account": ["account", "login", "password", "sign in", "delete account", "profile"],
            "privacy": ["privacy", "data", "gdpr", "personal information", "delete conversation"],
            "security": ["security", "fraud", "unauthorized", "stolen", "breach"],
            "screen": ["assessment", "test", "candidate", "invite", "proctoring", "screen", "recruit"],
            "community": ["community", "forum", "profile", "developer"],
            "travel_support": ["traveller", "traveler", "cheque", "travel"],
            "card_services": ["card", "atm", "contactless", "pin", "chip"],
            "general_support": [],
        }

        for area, signals in AREA_SIGNALS.items():
            if any(s in name or s in text for s in signals):
                return area
        return "general_support"

    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        company: Optional[str] = None,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Return top_k relevant chunks for the given text.

        Each result dict:
          {
            "document": str,
            "company": str,
            "filename": str,
            "product_area": str,
            "distance": float,  # cosine distance (lower = more similar)
          }
        """
        query_embedding = self.model.encode(text).tolist()

        # Build optional company filter
        where = None
        if company and company.lower() in VALID_COMPANIES:
            where = {"company": company.lower()}

        # Fix #8: don't use collection.count() as n_results cap when a where filter is
        # active — count() returns the TOTAL collection size, not the filtered size,
        # so ChromaDB throws ValueError when the company has fewer than top_k chunks.
        # Fix #9: if the company-filtered query returns nothing, retry without filter.
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            if where:
                print(f"[retriever] Company-filtered query failed ({e}) — retrying without filter…")
                try:
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(top_k, self.collection.count()),
                        include=["documents", "metadatas", "distances"],
                    )
                    where = None  # mark that we fell back
                except Exception as e2:
                    print(f"[retriever] Fallback query also failed: {e2}")
                    return []
            else:
                print(f"[retriever] Query error: {e}")
                return []

        # Fix #9 continued: if filtered query succeeded but returned zero docs, retry globally
        if where and (not results["documents"] or not results["documents"][0]):
            print("[retriever] Company filter returned 0 results — retrying without filter…")
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, self.collection.count()),
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as e:
                print(f"[retriever] Fallback query failed: {e}")
                return []

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "document": doc,
                "company": meta.get("company", "unknown"),
                "filename": meta.get("filename", ""),
                "product_area": meta.get("product_area", "general_support"),
                "distance": round(dist, 4),
            })

        return chunks

    def format_chunks_for_prompt(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a string for the LLM prompt."""
        if not chunks:
            return "No relevant documentation found in the support corpus."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Corpus excerpt {i} — {chunk['company'].title()} / {chunk['filename']}]\n"
                f"{chunk['document']}"
            )
        return "\n\n---\n\n".join(parts)
