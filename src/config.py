import os

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Translation Models
    MODEL_EN_VI = "Helsinki-NLP/opus-mt-en-vi"
    MODEL_VI_EN = "Helsinki-NLP/opus-mt-vi-en"
    
    # LLM for RAG
    LLM_MODEL = "Qwen/Qwen2-0.5B-Instruct"
    
    # Embedding
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Paths
    DATASET_DIR = os.path.join(BASE_DIR, "data", "processed")
    VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "vector_store")
    LOG_DB_PATH = os.path.join(BASE_DIR, "data", "translation_logs.db")
    
    # Training Config
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 16
    NUM_EPOCHS = 3
    MAX_LENGTH = 256
    
    # Domains
    SUPPORTED_DOMAINS = ["IT"]