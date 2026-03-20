from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from src.config import Config

try:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    ONNX_SUPPORTED = True
except ImportError:
    ONNX_SUPPORTED = False
    ORTModelForSeq2SeqLM = type("ORTModelForSeq2SeqLM_Stub", (), {})

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

class BaselineTranslator:
    def __init__(self):
        print("Loading models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}        # Caches models by key: domain_source_target
        self.tokenizers = {}
        self.pt_base_models = {} # Caches raw PyTorch base models to attach multiple LoRA adapters
        
        # Preload general models to be ready
        self._load_model("General", "en", "vi")
        self._load_model("General", "vi", "en")

    def _load_model(self, domain, source_lang, target_lang):
        key = f"{domain}_{source_lang}_{target_lang}"
        if key in self.models:
            return

        # 1. Ưu tiên load model ONNX nếu có (thường do file optimize.py tạo ra)
        onnx_dir = os.path.join(Config.BASE_DIR, "models", "onnx", f"{source_lang}-{target_lang}", domain)
        if ONNX_SUPPORTED and os.path.exists(onnx_dir):
            print(f"🚀 Tăng tốc bằng ONNX Runtime cho {key}...")
            self.tokenizers[key] = AutoTokenizer.from_pretrained(onnx_dir)
            self.models[key] = ORTModelForSeq2SeqLM.from_pretrained(onnx_dir)
            return

        # 2. Xử lý tải bằng PyTorch truyền thống
        base_key = f"General_{source_lang}_{target_lang}"
        if domain == "General":
            model_name = Config.MODEL_EN_VI if source_lang == "en" else Config.MODEL_VI_EN
            print(f"⏳ Tải Base Model PyTorch cho {key}...")
            self.tokenizers[key] = AutoTokenizer.from_pretrained(model_name)
            self.models[key] = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.pt_base_models[f"{source_lang}_{target_lang}"] = self.models[key]
        else:
            # Domain cần PyTorch Base bọc bởi LoRA
            pt_base_key = f"{source_lang}_{target_lang}"
            if pt_base_key not in self.pt_base_models:
                # Nếu General đã là ONNX, ta chưa có mô hình PyTorch trong RAM để cắm Adapter
                print("⏳ Tải PyTorch Base Model để sử dụng cho LoRA...")
                model_name = Config.MODEL_EN_VI if source_lang == "en" else Config.MODEL_VI_EN
                pt_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
                self.pt_base_models[pt_base_key] = pt_model
                self.tokenizers[pt_base_key] = AutoTokenizer.from_pretrained(model_name)
            
            base_model = self.pt_base_models[pt_base_key]
            self.tokenizers[key] = self.tokenizers[pt_base_key]
            
            # Khởi tạo hoặc Load LoRA
            adapter_path = os.path.join(getattr(Config, "ADAPTERS_DIR", os.path.join(Config.BASE_DIR, "models", "adapters")), f"{source_lang}-{target_lang}", domain)
            if os.path.exists(adapter_path) and PeftModel is not None:
                if not isinstance(base_model, PeftModel):
                    print(f"🔌 Bọc PeftModel và tải Adapter {domain}...")
                    base_model = PeftModel.from_pretrained(base_model, adapter_path, adapter_name=domain)
                    self.pt_base_models[pt_base_key] = base_model # Update type
                elif domain not in base_model.peft_config:
                    print(f"🔌 Bổ sung thêm Adapter {domain} vào bộ nhớ...")
                    base_model.load_adapter(adapter_path, adapter_name=domain)
                
                self.models[key] = base_model
            else:
                print(f"⚠️ Adapter cho {domain} không tồn tại. Tạm thời dùng General.")
                if base_key not in self.models:
                    self._load_model("General", source_lang, target_lang)
                self.models[key] = self.models[base_key]
                self.tokenizers[key] = self.tokenizers.get(base_key, self.tokenizers[pt_base_key])

    def _translate(self, text, source_lang, target_lang, domain):
        key = f"{domain}_{source_lang}_{target_lang}"
        if key not in self.models:
            self._load_model(domain, source_lang, target_lang)
            
        model = self.models[key]
        tokenizer = self.tokenizers[key]

        # Kích hoạt đúng Adapter nếu mô hình này là PeftModel
        context_manager = None
        if isinstance(model, PeftModel):
            if domain in model.peft_config:
                model.set_adapter(domain)
            else:
                context_manager = model.disable_adapter()

        if context_manager is None:
            from contextlib import nullcontext
            context_manager = nullcontext()

        with context_manager:
            device = "cuda" if torch.cuda.is_available() and not ONNX_SUPPORTED else "cpu"
            # ONNX optimum generates inputs transparently, but pt inputs need pushing to device
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            if not (ONNX_SUPPORTED and isinstance(model, ORTModelForSeq2SeqLM)):
                inputs = inputs.to(self.device)
                
            outputs = model.generate(**inputs, max_length=128)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate(self, text: str, source_lang: str, target_lang: str, domain: str = "General") -> str:
        if source_lang == "en" and target_lang == "vi":
            return self._translate(text, source_lang, target_lang, domain)

        elif source_lang == "vi" and target_lang == "en":
            return self._translate(text, source_lang, target_lang, domain)

        else:
            raise ValueError(f"Unsupported: {source_lang}-{target_lang}")