from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import time

from src.models.baseline import BaselineTranslator
from src.rag.pipeline import RAGTranslator
from src.config import Config

app = FastAPI(title="AI Translator Service API", version="1.0.0")

# Khởi tạo mô hình lúc start server
print("Khởi tạo API Server...")
baseline_translator = BaselineTranslator()
rag_translator = RAGTranslator()

class TranslateRequest(BaseModel):
    text: str
    source_lang: str # 'en' or 'vi'
    target_lang: str # 'en' or 'vi'
    domain: str = "General"
    mode: str = "Baseline" # 'Baseline' hoặc 'RAG'

class TranslateResponse(BaseModel):
    translated_text: str
    latency_ms: float
    mode: str
    domain: str

def save_translation_log(text, translated_text, mode, domain, latency):
    """Giả lập việc lưu Log Translation vào Database cho Admin theo FR-12"""
    # Thực tế sẽ dùng SQLAlchemy / MongoDB để insert
    # print(f"Logged to DB: {mode} | {domain} | Latency: {latency:.2f}ms")
    pass

@app.post("/api/v1/translate", response_model=TranslateResponse)
async def translate_endpoint(request: TranslateRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    
    # Logic dịch dựa theo chế độ
    try:
        if request.mode.upper() == "RAG":
            translated_text = rag_translator.translate_with_context(
                request.text, request.domain, request.source_lang, request.target_lang
            )
        else:
            translated_text = baseline_translator.translate(
                request.text, request.source_lang, request.target_lang
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    latency_ms = (time.time() - start_time) * 1000
    
    # Lưu log dưới nền để không block response cho client
    background_tasks.add_task(
        save_translation_log, 
        request.text, translated_text, request.mode, request.domain, latency_ms
    )
    
    return TranslateResponse(
        translated_text=translated_text,
        latency_ms=latency_ms,
        mode=request.mode,
        domain=request.domain
    )
