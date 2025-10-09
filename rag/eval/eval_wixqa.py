import argparse
import json
import random

from datasets import load_dataset
import numpy as np

from retrievers import rag_system_predict, TFIDFRetriever

from metrics import exact_match, recall_at_k, mrr_at_k, ndcg_at_k, hits_at_k, rougeL_f1, try_bertscore

# Evaluation loop
def evaluate_split(split_name: str, k: int, use_bertscore: bool, limit: int = None):
    qa = load_dataset("Wix/WixQA", split="train", name=split_name)
    print(f"[info] Loaded {split_name}: {len(qa)} examples")

    kb = load_dataset("Wix/WixQA", name="wix_kb_corpus")["train"]
    kb_lookup = {row["id"]: row for row in kb}
    _fallback_retriever = TFIDFRetriever({row["id"]: row["contents"] for row in kb})

    # subsample
    idxs = list(range(len(qa)))
    if limit:
        random.seed(0); random.shuffle(idxs)
        idxs = idxs[:limit]

    # accumulators
    r_at1 = r_at5 = r_at10 = 0.0
    mrr10 = 0.0
    ndcg10 = 0.0
    gen_em = []
    gen_rouge = []
    bert_pairs = []  

    for i in idxs:
        ex = qa[i]
        q = ex["question"]
        ref_answer = ex["answer"]
        gold_ids = ex["article_ids"]

        pred_answer, pred_ids = rag_system_predict(_fallback_retriever, q, k, kb_lookup)

        # retrieval metrics
        r_at1 += recall_at_k(pred_ids, gold_ids, 1)
        r_at5 += recall_at_k(pred_ids, gold_ids, 5)
        r_at10 += recall_at_k(pred_ids, gold_ids, 10)
        mrr10 += mrr_at_k(pred_ids, gold_ids, 10)
        ndcg10 += ndcg_at_k(pred_ids, gold_ids, 10)

        # generation metrics
        # NOTE: Alternatively you can have an LLM score the generated text based on the reference answers
        if pred_answer is None:
            pred_answer = ""
        gen_em.append(exact_match(pred_answer, ref_answer))
        gen_rouge.append(rougeL_f1(pred_answer, ref_answer))
        if use_bertscore:
            bert_pairs.append((pred_answer, ref_answer))

    n = len(idxs)
    results = {
        "split": split_name,
        "n": n,
        "retrieval": {
            "hit@1": r_at1 / n,
            "hit@5": r_at5 / n,
            "hit@10": r_at10 / n,
            "mrr@10": mrr10 / n,
            "ndcg@10": ndcg10 / n,
        },
        "generation": {
            "exact_match": float(np.mean(gen_em)),
            "rougeL_f1": float(np.mean(gen_rouge)),
        }
    }

    if use_bertscore and bert_pairs:
        bs = try_bertscore([p for p,_ in bert_pairs], [r for _,r in bert_pairs])
        if bs is not None:
            results["generation"]["bertscore_f1"] = float(np.mean(bs))

    print(json.dumps(results, indent=2))
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="Top-K docs to retrieve/evaluate")
    #ap.add_argument("--splits", nargs="+", default=["wixqa_expertwritten", "wixqa_simulated", "wixqa_synthetic"])
    # NOTE: Removing wixqa_synthetic for tests because it's very slow
    ap.add_argument("--splits", nargs="+", default=["wixqa_expertwritten", "wixqa_simulated"])
    ap.add_argument("--limit", type=int, default=None, help="Subsample for quick runs")
    ap.add_argument("--bertscore", action="store_true")
    args = ap.parse_args()

    all_res = []
    for s in args.splits:
        all_res.append(evaluate_split(s, k=args.k, use_bertscore=args.bertscore, limit=args.limit))

    # Simple table
    print("\n=== Summary ===")
    for r in all_res:
        m = r["retrieval"]; g = r["generation"]
        line = f"{r['split']:>18} | hit@1 {m['hit@1']:.3f} hit@5 {m['hit@5']:.3f} hit@10 {m['hit@10']:.3f} mrr@10 {m['mrr@10']:.3f} ndcg@10 {m['ndcg@10']:.3f} | EM {g['exact_match']:.3f} ROUGE-L {g['rougeL_f1']:.3f}"
        if "bertscore_f1" in g:
            line += f" BERTScore {g['bertscore_f1']:.3f}"
        print(line)

if __name__ == "__main__":
    main()

