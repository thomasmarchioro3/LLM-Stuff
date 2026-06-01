
from langchain_ollama import OllamaEmbeddings

def get_embeddings():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore")

    embeddings = get_embeddings()
    print(embeddings.embed_query("The fish twisted and turned").__len__())
