import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import Config

class VectorStore:
    def __init__(self):
        """Khởi tạo cơ sở dữ liệu Vector để xử lý RAG"""
        print(f"Loading Embedding Model: {Config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        self.indices = {}      # Lưu Faiss Index theo từng domain
        self.documents = {}    # Lưu trữ mapping văn bản
        
    def load_domain(self, domain: str):
        """Khởi tạo hoặc tải Index cho một Domain cụ thể"""
        self.indices[domain] = faiss.IndexFlatL2(self.dimension)
        self.documents[domain] = []
        
    def add_documents(self, domain: str, texts: list):
        """Thêm các thuật ngữ, tài liệu chuyên ngành vào Vector Store"""
        if domain not in self.indices:
            self.load_domain(domain)
            
        embeddings = self.embedding_model.encode(texts)
        self.indices[domain].add(np.array(embeddings, dtype=np.float32))
        self.documents[domain].extend(texts)
        
    def search(self, domain: str, query: str, top_k: int = 3) -> list:
        """Tìm kiếm các câu/thuật ngữ tương đồng nhất với truy vấn"""
        if domain not in self.indices or self.indices[domain].ntotal == 0:
            return []
            
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.indices[domain].search(np.array(query_embedding, dtype=np.float32), top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.documents[domain]):
                results.append(self.documents[domain][idx])
        return results
    def save(self, domain: str):
    path = os.path.join(Config.VECTOR_DB_PATH, f"{domain}.index")
    os.makedirs(Config.VECTOR_DB_PATH, exist_ok=True)
    faiss.write_index(self.indices[domain], path)

def load(self, domain: str):
    path = os.path.join(Config.VECTOR_DB_PATH, f"{domain}.index")
    if os.path.exists(path):
        self.indices[domain] = faiss.read_index(path)
        self.documents[domain] = []  # cần load text riêng nếu có