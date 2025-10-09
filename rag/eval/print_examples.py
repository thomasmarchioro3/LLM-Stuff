from datasets import load_dataset
import random
import textwrap

from retrievers import TFIDFRetriever

# helper to pretty-print wrapped text
def wrap(text, width=90, indent=""):
    wrapped = textwrap.fill(text.strip(), width=width, initial_indent=indent, subsequent_indent=indent)
    return wrapped

# visualize n random examples
def visualize_examples(n=3, k=5):
    for idx in random.sample(range(len(qa)), n):
        q = qa[idx]["question"]
        gold_ids = qa[idx]["article_ids"]
        gold_titles = [kb_lookup[i][:80].replace("\n"," ")+"..." for i in gold_ids if i in kb_lookup]
        print("="*120)
        print(f"[{idx}] ❓ QUESTION:\n{wrap(q, indent='   ')}\n")
        print(f"✅ GOLD article_ids ({len(gold_ids)}): {gold_ids}")
        for t in gold_titles:
            print(f"   → {wrap(t, width=100)}")

        retrieved = retriever.search(q, k=k)
        print(f"\n🔍 TOP-{k} retrieved docs:")
        for rank, rid in enumerate(retrieved, 1):
            mark = "⭐" if rid in gold_ids else " "
            snippet = kb_lookup[rid][:180].replace("\n", " ")
            print(f" {mark}  {rank:>2}. ID={rid}")
            print(f"     {wrap(snippet, width=100, indent='     ')}\n")


if __name__ == "__main__":

    qa = load_dataset("Wix/WixQA", name="wixqa_expertwritten", split="train")
    kb = load_dataset("Wix/WixQA", name="wix_kb_corpus")["train"]

    # map IDs to text for easy lookup
    kb_lookup = {row["id"]: row["contents"] for row in kb}

    retriever = TFIDFRetriever(kb_lookup)

    visualize_examples(n=3, k=5)
