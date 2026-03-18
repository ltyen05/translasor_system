import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.config import Config

def export_to_onnx(model_name: str = Config.MODEL_EN_VI, output_dir: str = "./models/onnx/en-vi"):
    """
    Tối ưu hóa Inference bằng cách chuyển đổi mô hình PyTorch sang định dạng ONNX.
    ONNX (Open Neural Network Exchange) giúp giảm đáng kể Latency (đến 2-3x) 
    khi chạy Inference trên cả CPU và GPU.
    """
    print(f"Loading Base Model for ONNX Export: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Ví dụ input cho model học hình dạng tensor (tracing)
    dummy_input = tokenizer("This is a test sentence for ONNX export.", return_tensors="pt")
    
    onnx_path = os.path.join(output_dir, "model.onnx")
    print(f"Exporting model to {onnx_path}...")
    
    # Lưu ý: Với họ model encoder-decoder (như mBART, MarianMT), việc export phức tạp hơn
    # do có phần encoder và decoder riêng biệt. Ở đây dùng API chuẩn của PyTorch cho mục đích minh họa khái niệm.
    # Trong thực tế với HuggingFace, nên dùng thư viện Optimum: 
    # `optimum-cli export onnx --model Helsinki-NLP/opus-mt-en-vi models/onnx/en-vi`
    
    # Đoạn code dưới đây mô phỏng khái niệm tối ưu NFR-01 (Latency)
    try:
        # Example using torch.onnx.export (Giản lược cho Encoder)
        torch.onnx.export(
            model.get_encoder(),
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            os.path.join(output_dir, "encoder.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=14,
        )
        print("✅ ONNX Export Successful (Encoder simulated)!")
        print("Hint: Để dùng Optimum đầy đủ: pip install optimum[onnxruntime]")
    except Exception as e:
        print(f"Lỗi xuất ONNX (cần thư viện chuyên dụng cho Seq2Seq): {e}")

if __name__ == "__main__":
    # export_to_onnx()
    pass
