# pyright: basic
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore
from datapizza.type.type import Node

query = "Who/what is Adam? I think this is a very serious typo."

class MyDoclingParser(DoclingParser):
    def parse(self, text: str, metadata: dict) -> Node:
        return super().parse(text)

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

docling_parser = MyDoclingParser()
splitter = NodeSplitter(max_char=1000)
embedder = ChunkEmbedder(client=embedder_client)  # pyright: ignore

ingestion_pipeline = IngestionPipeline(
    modules=[
        docling_parser,
        splitter,
        embedder,
    ], # pyright: ignore
    vector_store=vectorstore,
    collection_name="my_documents"
)  

ingestion_pipeline.run("sample.pdf", metadata={"source": "user_upload"})

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
