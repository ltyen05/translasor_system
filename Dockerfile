# Sử dụng image Python làm base
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Khắc phục lỗi thiếu thư viện C++ cơ bản cho FAISS và thư viện Machine Learning
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Tạo các thư mục dữ liệu cần thiết
RUN mkdir -p /app/data/vector_store
RUN touch /app/data/translation_logs.db

# Copy mã nguồn dự án vào container
COPY src /app/src

# Expose port cho FastAPI
EXPOSE 8000

# Lệnh khởi chạy Server (Với uvicorn)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
