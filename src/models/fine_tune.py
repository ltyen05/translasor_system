import os
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from src.config import Config
from src.evaluation.metrics import calculate_bleu


def fine_tune_model(
    dataset_dir: str,
    domain: str = "IT",
    source_lang: str = "en",
    target_lang: str = "vi"
):
    model_name = Config.MODEL_EN_VI if source_lang == "en" else Config.MODEL_VI_EN
    output_dir = os.path.join(getattr(Config, "ADAPTERS_DIR", os.path.join(Config.BASE_DIR, "models", "adapters")), f"{source_lang}-{target_lang}", domain)
    print(f"🚀 Loading Base Model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Configure LoRA Adapter
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ✅ Load dataset (train / valid / test)
    dataset = load_dataset("csv", data_files={
        "train": os.path.join(dataset_dir, "train.csv"),
        "validation": os.path.join(dataset_dir, "valid.csv"),
        "test": os.path.join(dataset_dir, "test.csv")
    })

    print("📂 Dataset loaded!")

    # ✅ Preprocess
    def preprocess_function(examples):
        inputs = [f"{d}: {en}" for d, en in zip(examples["domain"], examples["en"])]
        targets = examples["vi"]

        model_inputs = tokenizer(
            inputs,
            max_length=Config.MAX_LENGTH,
            truncation=True,
            padding="max_length"
        )

        labels = tokenizer(
            text_target=targets,
            max_length=Config.MAX_LENGTH,
            truncation=True,
            padding="max_length"
        )

        labels_ids = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label]
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels_ids
        return model_inputs

    print("⚙️ Tokenizing dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # ✅ Metrics
    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu = calculate_bleu(decoded_preds, decoded_labels)
        return {"bleu": bleu}

    # ✅ Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",

        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.NUM_EPOCHS,

        weight_decay=0.01,
        predict_with_generate=False, # 🔴 TẮT CÁI NÀY: Dừng việc dịch thử toàn bộ Validation Set để tiết kiệm hàng giờ đồng hồ rác!
        generation_max_length=Config.MAX_LENGTH,

        fp16=False,  # ⚠️ set True nếu có GPU
        logging_steps=50,
        save_total_limit=2
    )

    # ✅ Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer
        # Đã gỡ bỏ compute_metrics để tránh báo lỗi khi predict_with_generate=False
    )

    print(" === TRAINING START ===")
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved at: {output_dir}")


if __name__ == "__main__":
    fine_tune_model(dataset_dir=Config.DATASET_DIR)