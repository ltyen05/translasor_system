import sacrebleu
from bert_score import score

def calculate_bleu(hypotheses: list, references: list) -> float:
    """
    Tính toán chỉ số BLEU score (Truyền thống, đo lường word overlap).
    :param hypotheses: List các bản dịch của hệ thống.
    :param references: List các bản dịch chuẩn (Ground truth).
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score

def calculate_bertscore(hypotheses: list, references: list, lang: str="vi") -> dict:
    """
    Tính toán chỉ số BERTScore (Semantic similarity sử dụng pre-trained language model).
    :param hypotheses: List các bản dịch của hệ thống.
    :param references: List các bản dịch chuẩn.
    :param lang: Ngôn ngữ đích để đánh giá.
    """
    # verbose=False để tránh in quá nhiều log
    P, R, F1 = score(hypotheses, references, lang=lang, verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }
