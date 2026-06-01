import warnings
warnings.filterwarnings("ignore")

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# Local modules
from utils.path import PDF_PATH, CHROMA_PATH
from embeddings import get_embeddings

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

TOP_K = 3

def naive_rag(model: OllamaLLM, db: Chroma, query: str, top_k: int) -> str:
    """
    Naive RAG (retrieval + generation) implemented according to https://arxiv.org/pdf/2312.10997
    The indexing step is performed by the `database.pdf2chroma` module.
    """

    # Retrieval
    relevant_chunks = db.similarity_search(query, k=top_k)
    # NOTE: It is possible to use `db.similarity_search_with_score(query, k=top_k)` to get also the scores
    # the scores can be used to filter out irrelevant chunks

    context = "\n\n----\n\n".join(map(lambda chunk: chunk.page_content, relevant_chunks))
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context, 
        question=query
    )

    print("="*32)
    print(f"Context: {context}")
    print("="*32)

    # Generation
    response_text = model.invoke(prompt)

    source_ids = list(map(lambda chunk: chunk.metadata["id"], relevant_chunks))

    response = f"Response: {response_text}\n\nSources: {source_ids}"

    return response


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, default=PDF_PATH)
    parser.add_argument("--chroma_path", type=str, default=CHROMA_PATH)
    parser.add_argument("--top_k", type=int, default=TOP_K)

    args = parser.parse_args()

    db = Chroma(
        persist_directory=args.chroma_path,
        embedding_function=get_embeddings(),
    )

    model = OllamaLLM(
        # endpoint="http://localhost:11434",
        model="llama3.2:1b"
    )

    while True:
        query: str = input("Enter your question (/bye to exit): ")
        if query.strip() in ["/bye", "/exit", "/quit"]:
            print("Bye!")
            break
        response = naive_rag(model, db, query, args.top_k)
        print(response)
