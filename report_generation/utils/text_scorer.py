from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

class TextScorer:
    def __init__(self):
        self.bleu_smooting = SmoothingFunction().method1  # Avoids zero scores for short texts
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def __call__(self, reference: str, hypothesis: str) -> tuple[float, float, float, float, float]:
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=self.bleu_smooting)
        rouge = self.rouge_scorer.score(reference, hypothesis)
        meteor = meteor_score([ref_tokens], hyp_tokens)

        return bleu, rouge['rouge1'].fmeasure, rouge['rouge2'].fmeasure, rouge['rougeL'].fmeasure, meteor