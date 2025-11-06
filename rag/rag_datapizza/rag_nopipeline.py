from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
#from datapizza.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore
from datapizza.type.type import Node

import os
import pickle

# used for caching OCR results
cache_file = "cache.pkl"
query = "Who/what is Adam? I think this is a very serious typo."

vectorstore = QdrantVectorstore(location=":memory:")
vectorstore.create_collection(
    "my_documents",
    vector_config=[VectorConfig(name="nomic-embed-text:latest", dimensions=768)]
)

embedder_client = OpenAIEmbedder(
    api_key="",
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434/v1",
)

parser = DoclingParser()
splitter = NodeSplitter(max_char=1000)
embedder = ChunkEmbedder(client=embedder_client)

if not os.path.exists(cache_file):
    doc = parser.parse(file_path="sample.pdf")
    chunks = splitter.split(doc)
    with open(cache_file, "wb") as f:
        pickle.dump(chunks, f)
else:
    with open(cache_file, "rb") as f:
        chunks = pickle.load(f)

embeddings = embedder.embed(chunks)

vectorstore.add(chunk=embeddings, collection_name="my_documents")

query_node = Node(content=query)
query_chunks = splitter.split(query_node)
query_vector = embedder.embed(query_chunks)[0].embeddings[0].vector


res = vectorstore.search(
    query_vector = query_vector,
    collection_name="my_documents",
    k=5,
)

print("Query: ", query)
print("Results:")
for r in res:
    print(r.text)
    print(r.metadata)
