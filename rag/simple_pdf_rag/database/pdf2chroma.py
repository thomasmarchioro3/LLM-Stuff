"""
Script for indexing a directory of PDF files into Chroma.

The indexing step is performed by the `database.pdf2chroma` module.

Usage (from the root directory):
```
python -m database.pdf2chroma [--args]
```

Args:
    --pdf_path: The path to the directory of PDF files.
    --chroma_path: The path to the directory where the Chroma database will be stored.
    --chunk_size: The maximum number of tokens in a chunk.
    --chunk_overlap: The number of tokens to overlap between two consecutive chunks.
"""

import os
import shutil
from pathlib import Path
from collections import Counter

# External libraries
# NOTE: langchain_community is being sunset, but there is no official replacement for PyPDFLoader (to my knowledge)
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local modules
from embeddings import get_embeddings
from utils.path import PDF_PATH, CHROMA_PATH

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 10


def get_documents(directory_path: str) -> list[Document]:
    docs = []
    # Use standard library to find files
    path = Path(directory_path)
    for pdf_file in path.glob("*.pdf"):
        # Instantiate the specific loader for the individual file
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())
    return docs


def split_documents_into_chunks(
        documents: list[Document], 
        chunk_size: int,
        chunk_overlap: int
    ) -> list[Document]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        )
    chunks = text_splitter.split_documents(documents)
    return chunks


def assign_chunk_ids(chunks: list[Document]) -> list[Document]:
    """
    Assign unique IDs to each chunk in a list of chunks.
    The IDs will be of the form `source:page:number`.

    Args:
        chunks: A list of Document objects.

    Returns:
        The same list of Document objects, but with unique IDs assigned to each
        chunk.
    """

    pages = list(map(lambda chunk: f"{chunk.metadata["source"]}:{chunk.metadata["page"]}", chunks))

    # TODO: Find a way to vectorize the ID assignment
    counter = Counter()
    for i, page in enumerate(pages):
        counter[page] += 1
        chunks[i].metadata["id"] = f"{page}:{counter[page]}"

    return chunks


def add_chunks_to_chroma(db: Chroma, chunks: list[Document]) -> list[Document]:

    chunks = assign_chunk_ids(chunks) 

    existing_items = db.get(
        include=[],  # do not include any data (IDs are always included)
    )

    if len(existing_items) > 0:
        existing_ids = list(set(existing_items["ids"]))
        chunks = list(filter(lambda chunk: chunk.metadata["id"] not in existing_ids, chunks))
    
    if len(chunks) > 0:
        db.add_documents(chunks, ids=list(map(lambda chunk: chunk.metadata["id"], chunks)))

    return chunks


def clear_db(persist_directory: str | Path) -> None:
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, default=PDF_PATH)
    parser.add_argument("--chroma_path", type=str, default=CHROMA_PATH)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)

    args = parser.parse_args()


    db = Chroma(
        persist_directory=args.chroma_path,
        embedding_function=get_embeddings(),
    )

    documents = get_documents(args.pdf_path)
    chunks = split_documents_into_chunks(documents, args.chunk_size, args.chunk_overlap)
    print(f"Number of extracted chunks: {len(chunks)}")
    uploaded_chunks = add_chunks_to_chroma(db, chunks)
    print(f"Number of uploaded chunks: {len(uploaded_chunks)}")
