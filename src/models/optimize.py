import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.config import Config

try:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from peft import PeftModel
except ImportError:
    print("Vui lòng cài đặt optimum và peft: pip install optimum[onnxruntime] peft")
    exit(1)

def export_model_to_onnx(domain: str = "General", source_lang: str = "en", target_lang: str = "vi"):
    """
    Hợp nhất LoRA Adapter (nếu có) vào Base Model, và xuất ra định dạng ONNX 
    để tối đa hoá tốc độ nội suy.
    """
    model_name = Config.MODEL_EN_VI if source_lang == "en" else Config.MODEL_VI_EN
    
    onnx_output_dir = os.path.join(Config.BASE_DIR, "models", "onnx", f"{source_lang}-{target_lang}", domain)
    os.makedirs(onnx_output_dir, exist_ok=True)
    
    print(f"⏳ Đang xử lý model {source_lang}-{target_lang} cho domain '{domain}'...")
    
    # 1. Load Base Model bằng PyTorch
    print(" - Load Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 2. Xử lý LoRA nếu domain không phải là General
    if domain != "General":
        adapter_path = os.path.join(getattr(Config, "ADAPTERS_DIR", os.path.join(Config.BASE_DIR, "models", "adapters")), f"{source_lang}-{target_lang}", domain)
        if os.path.exists(adapter_path):
            print(f" - Load LoRA Adapter từ {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path)
            print(" - Hợp nhất (Merge) trọng số LoRA vào Base Model...")
            model = model.merge_and_unload()
        else:
            print(f" ⚠️ Không tìm thấy Adapter cho {domain}, hệ thống sẽ lưu file ONNX cấu trúc mạng Base!")
    else:
        print(" - Bỏ qua LoRA (Đang xử lý Base Model - General)")

    # 3. Lưu model tạm ra ổ cứng để Optimum có thể convert thành ONNX
    temp_dir = os.path.join(Config.BASE_DIR, "models", "temp_merge")
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)
    
    # 4. Xuất sang ONNX (Export)
    print(f" - Bắt đầu quá trình Export sang ONNX (Mất khoảng vài phút)...")
    ort_model = ORTModelForSeq2SeqLM.from_pretrained(temp_dir, export=True)
    ort_model.save_pretrained(onnx_output_dir)
    tokenizer.save_pretrained(onnx_output_dir)
    
    print(f"✅ Đã xuất ONNX thành công tại: {onnx_output_dir}!\n")

if __name__ == "__main__":
    print("=== CÔNG CỤ TỐI ƯU HOÁ ONNX ===")
    
    # Tối ưu mặc định model General
    export_model_to_onnx(domain="General", source_lang="en", target_lang="vi")
    # export_model_to_onnx(domain="General", source_lang="vi", target_lang="en")
    
    # Ở đây tự động quét các domain bạn đã train:
    for d in Config.SUPPORTED_DOMAINS:
        adapter_en_vi = os.path.join(getattr(Config, "ADAPTERS_DIR", os.path.join(Config.BASE_DIR, "models", "adapters")), "en-vi", d)
        if os.path.exists(adapter_en_vi):
            export_model_to_onnx(domain=d, source_lang="en", target_lang="vi")
