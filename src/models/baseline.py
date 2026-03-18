from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import Config
import torch

class BaselineTranslator:
    def __init__(self):
        print("Loading models...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.en_vi_tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_EN_VI)
        self.en_vi_model = AutoModelForSeq2SeqLM.from_pretrained(
            Config.MODEL_EN_VI
        ).to(self.device)

        self.vi_en_tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_VI_EN)
        self.vi_en_model = AutoModelForSeq2SeqLM.from_pretrained(
            Config.MODEL_VI_EN
        ).to(self.device)

    def _translate(self, text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        outputs = model.generate(**inputs, max_length=128)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == "en" and target_lang == "vi":
            return self._translate(text, self.en_vi_tokenizer, self.en_vi_model)

        elif source_lang == "vi" and target_lang == "en":
            return self._translate(text, self.vi_en_tokenizer, self.vi_en_model)

        else:
            raise ValueError(f"Unsupported: {source_lang}-{target_lang}")