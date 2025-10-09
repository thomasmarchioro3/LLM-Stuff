import re
import math
from typing import List

from rouge_score import rouge_scorer

def dcg(rels):
    return sum((rel / math.log2(i+2)) for i, rel in enumerate(rels))

def ndcg_at_k(pred_ids: List[str], gold_ids: List[str], k: int = 10) -> float:
    gains = [1.0 if d in set(gold_ids) else 0.0 for d in pred_ids[:k]]
    idcg = dcg(sorted([1.0]*min(len(gold_ids), k), reverse=True))
    return dcg(gains)/idcg if idcg > 0 else 0.0

def mrr_at_k(pred_ids: List[str], gold_ids: List[str], k: int = 10) -> float:
    gold = set(gold_ids)
    for i, d in enumerate(pred_ids[:k]):
        if d in gold:
            return 1.0/(i+1)
    return 0.0

def recall_at_k(pred_ids: List[str], gold_ids: List[str], k: int = 10) -> float:
    return 1.0 if any(d in set(gold_ids) for d in pred_ids[:k]) else 0.0

def hits_at_k(pred_ids: List[str], gold_ids: List[str], k: int = 5) -> float:
    return recall_at_k(pred_ids, gold_ids, k)

# Generation metrics (EM, ROUGE-L; BERTScore optional)
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def normalize_text(s):
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(a, b):
    return 1.0 if normalize_text(a) == normalize_text(b) else 0.0

def rougeL_f1(pred, ref):
    return rouge.score(ref, pred)["rougeL"].fmeasure

def try_bertscore(cands, refs):
    try:
        import bert_score
        P, R, F = bert_score.score(cands, refs, lang="en", rescale_with_baseline=True)
        return F.numpy().tolist()
    except Exception as e:
        print("[warn] BERTScore not available:", e)
        return None

