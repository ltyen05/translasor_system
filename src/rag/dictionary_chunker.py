import json
from src.rag.vector_store import VectorStore

class DictionaryChunker:
    """
    Class hỗ trợ đọc các file từ điển chuyên ngành hoặc tài liệu 
    và phân rã (chunking) thành các câu/thuật ngữ để đưa vào VectorStore.
    """
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
    def process_json_dictionary(self, file_path: str, domain: str):
        """
        Xử lý từ điển định dạng JSON:
        [
            {"term": "Load Balancer", "meaning": "Bộ cân bằng tải", "context": "Dùng để phân phối traffic."}
        ]
        """
        print(f"Loading dictionary for domain '{domain}' from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            chunks = []
            for item in data:
                term = item.get("term", "")
                meaning = item.get("meaning", "")
                context = item.get("context", "")
                
                # Tạo một đoạn text giàu ngữ nghĩa cho Vector Store
                chunk = f"Thuật ngữ: {term}. Nghĩa Tiếng Việt: {meaning}. Ngữ cảnh/Ví dụ: {context}"
                chunks.append(chunk)
                
            if chunks:
                self.vector_store.add_documents(domain, chunks)
                print(f"Added {len(chunks)} chunks to '{domain}' vector store.")
                
        except Exception as e:
            print(f"Error processing dictionary: {e}")

if __name__ == "__main__":
    # Mock usage
    # store = VectorStore()
    # chunker = DictionaryChunker(store)
    # chunker.process_json_dictionary("data/dict_medical.json", "Medical")
    pass
