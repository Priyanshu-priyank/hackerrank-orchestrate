import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_PERSIST_PATH, EMBEDDING_MODEL, MAX_CHUNK_TOKENS


class Retriever:
    def __init__(self, corpus_path):
        self.corpus_path = Path(corpus_path).resolve()
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
        self.collection = self.client.get_or_create_collection(
            "support_corpus",
            metadata={"hnsw:space": "cosine"},
        )
        if self.collection.count() == 0:
            self._index_corpus()

    def _chunk_text(self, text, max_tokens=MAX_CHUNK_TOKENS):
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""

        for paragraph in paragraphs:
            combined = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(combined.split()) > max_tokens and current:
                if len(current.strip()) >= 30:
                    chunks.append(current.strip())
                current = paragraph
            else:
                current = combined

        if len(current.strip()) >= 30:
            chunks.append(current.strip())
        return chunks

    def _index_corpus(self):
        files = []
        for company in ("hackerrank", "claude", "visa"):
            company_dir = self.corpus_path / company
            if company_dir.exists():
                files.extend(company_dir.rglob("*.md"))
                files.extend(company_dir.rglob("*.txt"))

        documents = []
        ids = []
        metadatas = []

        for file_path in sorted(files):
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            company = file_path.relative_to(self.corpus_path).parts[0].lower()
            for i, chunk in enumerate(self._chunk_text(text)):
                documents.append(chunk)
                ids.append(self._make_id(file_path, i))
                metadatas.append(
                    {
                        "company": company,
                        "filename": file_path.name,
                        "product_area": file_path.stem,
                        "chunk_index": i,
                    }
                )

        for start in range(0, len(documents), 64):
            batch_docs = documents[start : start + 64]
            embeddings = self.model.encode(batch_docs).tolist()
            self.collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                ids=ids[start : start + 64],
                metadatas=metadatas[start : start + 64],
            )

    def query(self, text, company=None, top_k=3):
        query_text = (text or "").strip()
        if not query_text:
            return [], []

        try:
            embedding = self.model.encode(query_text).tolist()
            company_value = (company or "").strip().lower()
            if company_value and company_value != "none" and company_value != "unknown":
                documents, metadatas = self._query_with_embedding(
                    embedding,
                    top_k=top_k,
                    where={"company": company_value},
                )
                if documents:
                    return documents, metadatas

            return self._query_with_embedding(embedding, top_k=top_k, where=None)
        except (IndexError, ValueError):
            return [], []

    def _query_with_embedding(self, embedding, top_k=3, where=None):
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas"],
        }
        if where:
            kwargs["where"] = where
        results = self.collection.query(**kwargs)
        try:
            return results["documents"][0], results["metadatas"][0]
        except (IndexError, KeyError, TypeError):
            return [], []

    def _make_id(self, file_path, chunk_index):
        raw = f"{file_path.as_posix()}::{chunk_index}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()
