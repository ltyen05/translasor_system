# AI Translator System - Core Engine

Đây là source code khung (boilerplate) cho các kỹ sư AI triển khai hệ thống AI Translator (Anh ↔ Việt).

## 🗂 Cấu trúc thư mục định hướng AI
- `/src/models/`: Cấu hình Baseline Translation Models (mBART, NLLB, OPUS-MT).
- `/src/rag/`: Hệ thống Retrieval-Augmented Generation bao gồm chia vector, chunking từ điển và pipeline LLM.
- `/src/evaluation/`: Các code đánh giá tự động (BLEU, BERTScore).
- `/src/api/`: Model Serving API qua FastAPI.

## 🚀 Các bước khởi chạy môi trường
```bash
# 1. Cài đặt thư viện Python
pip install -r requirements.txt

# 2. Chạy Serving API (Phục vụ Web/Mobile App)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
