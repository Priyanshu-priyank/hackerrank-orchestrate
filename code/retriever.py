import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_PERSIST_PATH, CORPUS_PATH, MAX_CHUNK_TOKENS

class Retriever:
    def __init__(self):
        print(f"Loading SentenceTransformer model: {EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
        self.collection = self.client.get_or_create_collection(
            name="support_corpus",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Only index if empty (avoid re-indexing every run)
        if self.collection.count() == 0:
            print("ChromaDB is empty. Indexing corpus...")
            self._index_corpus()
        else:
            print(f"ChromaDB already initialized with {self.collection.count()} chunks.")

    def _chunk_text(self, text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[str]:
        paragraphs = text.split('\n\n')
        chunks = []
        current = ""
        for para in paragraphs:
            combined_len = len(current.split()) + len(para.split())
            if combined_len > max_tokens and current:
                chunks.append(current.strip())
                current = para
            else:
                current = (current + "\n\n" + para).strip()
        if current:
            chunks.append(current.strip())
        return [c for c in chunks if len(c.strip()) > 30]

    def _index_corpus(self):
        docs = []
        metas = []
        ids = []
        global_idx = 0
        
        # Define companies mapping based on folder names
        companies = ["hackerrank", "claude", "visa"]
        
        for company in companies:
            folder_path = os.path.join(CORPUS_PATH, company)
            if not os.path.exists(folder_path):
                print(f"Warning: Path {folder_path} does not exist.")
                continue
                
            # Iterate over markdown/txt files
            for file_path in glob.glob(os.path.join(folder_path, "**", "*.*"), recursive=True):
                if file_path.endswith((".md", ".txt", ".csv")):
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        
                    filename = os.path.basename(file_path)
                    
                    # extract product area from directory name
                    product_area = os.path.basename(os.path.dirname(file_path))
                    if product_area == company:
                        product_area = "general"
                        
                    chunks = self._chunk_text(content)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        docs.append(chunk)
                        metas.append({
                            "company": company.lower(),
                            "filename": filename,
                            "product_area": product_area.lower(),
                            "chunk_index": chunk_idx
                        })
                        ids.append(f"{company}_{filename}_{chunk_idx}")
                        global_idx += 1

        if not docs:
            print("No documents found to index.")
            return

        print(f"Adding {len(docs)} chunks to ChromaDB...")
        
        # Batch insert
        batch_size = 500
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            batch_metas = metas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            embeddings = self.model.encode(batch_docs).tolist()
            
            self.collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_metas,
                ids=batch_ids
            )
        print("Indexing complete.")

    def query(self, text: str, company: str = None, top_k: int = 3):
        # Truncate very long issues
        words = text.split()
        if len(words) > 1000:
            text = " ".join(words[:1000])
            
        embedding = self.model.encode(text).tolist()
        
        where_filter = None
        if company and company.lower() != "none":
            where_filter = {"company": company.lower()}
            
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                return [], []
                
            return results["documents"][0], results["metadatas"][0]
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return [], []
