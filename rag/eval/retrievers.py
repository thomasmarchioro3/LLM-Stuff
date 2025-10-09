import re
from typing import List, Tuple, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def simple_extractive_answer(question: str, contexts: list[str], max_chars: int = 512) -> str:
    # gather sentences from contexts
    sentences = []
    for c in contexts:
        if not c:
            continue
        c = c.replace("\n", " ").strip()
        sentences += re.split(r"(?<=[.!?])\s+", c)
    if not sentences:
        return ""

    # TF-IDF over question + candidate sentences
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_df=0.95)
    X = vec.fit_transform([question] + sentences)
    qv, sv = X[0:1], X[1:]
    sims = cosine_similarity(qv, sv)[0]
    best = sentences[int(sims.argmax())].strip()

    # clip long rambly sentences (KB can be verbose)
    return best[:max_chars]

def rag_system_predict(retriever, query: str, k: int, kb_lookup: dict) -> tuple[str, list[str]]:
    """
    Minimal plug-in RAG:
      1) retrieve top-k doc IDs via the built-in TF-IDF retriever
      2) build a short context set (top 5 docs)
      3) pick the single most similar sentence to the query
    """
    retrieved_ids = retriever.search(query, k=k)
    contexts = [kb_lookup[i]["contents"] for i in retrieved_ids[:5] if i in kb_lookup]
    answer = simple_extractive_answer(query, contexts)
    return answer, retrieved_ids



# ------- Minimal TF-IDF fallback retriever (for sanity-check baseline) -------
class TFIDFRetriever:
    def __init__(self, kb_docs: Dict[str,str]):
        self.ids = list(kb_docs.keys())
        self.texts = [kb_docs[i] for i in self.ids]
        self.vec = TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))
        self.M = self.vec.fit_transform(self.texts)

    def search(self, query: str, k: int = 20) -> List[str]:
        q = self.vec.transform([query])
        sims = cosine_similarity(q, self.M)[0]
        top = np.argsort(-sims)[:k]
        return [self.ids[i] for i in top]


