import json
from src.evaluation.metrics import calculate_bleu, calculate_bertscore
from src.models.baseline import BaselineTranslator

def run_benchmark(test_file: str, domain: str):
    """
    Tự động chạy đánh giá mô hình trên một Test Set và in ra kết quả.
    """
    print(f"Running benchmark for Domain: {domain}...")
    
    # 1. Tải mô hình
    translator = BaselineTranslator()
    
    # 2. Tải Test Set (File JSON Lines giống tập train)
    references = []
    inputs = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            inputs.append(data['translation']['en'])
            references.append(data['translation']['vi'])
            
    # Giới hạn số câu test để demo nhanh
    inputs = inputs[:20]
    references = references[:20]
    
    # 3. Chạy dịch tự động (Inference)
    print(f"Translating {len(inputs)} sentences...")
    hypotheses = []
    for text in inputs:
        # Mocking translation call
        res = translator.translate(text, source_lang="en", target_lang="vi")
        hypotheses.append(res)
        
    # 4. Tính toán Metrics
    print("Calculating Metrics...")
    bleu_score = calculate_bleu(hypotheses, references)
    bert_results = calculate_bertscore(hypotheses, references, lang="vi")
    
    print("\n" + "="*40)
    print(f"Benchmark Results - {domain} (EN -> VI)")
    print("="*40)
    print(f"BLEU Score : {bleu_score:.2f}")
    print(f"BERT P     : {bert_results['precision']:.4f}")
    print(f"BERT R     : {bert_results['recall']:.4f}")
    print(f"BERT F1    : {bert_results['f1']:.4f}")
    print("="*40)

if __name__ == "__main__":
    # run_benchmark("data/cleaned_dataset_it.json", "IT")
    pass
