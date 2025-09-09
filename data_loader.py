"""Load text/PDF files from support_docs/ into a Chroma vector store.
Usage: python data_loader.py
"""
import os
from pathlib import Path
from typing import List
import chromadb
from chromadb.config import Settings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

DATA_DIR = Path(__file__).parent / "support_docs"
PERSIST_DIR = Path(__file__).parent / ".chromadb"

def load_documents_from_dir(dir_path: Path) -> List[Document]:
    docs = []
    for p in dir_path.glob("**/*"):
        if p.is_file():
            if p.suffix.lower() in [".txt", ".md"]:
                loader = TextLoader(str(p), encoding='utf-8')
                docs.extend(loader.load())
            elif p.suffix.lower() in [".pdf"]:
                loader = PyPDFLoader(str(p))
                docs.extend(loader.load())
    return docs

def main():
    if not DATA_DIR.exists():
        print("support_docs/ is empty — drop PDFs or text files there and run this script.")
        return
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(PERSIST_DIR)))
    collection = client.get_or_create_collection(name="support_collection")
    # Prepare embeddings
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable before running.")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    # Load and split documents
    docs = load_documents_from_dir(DATA_DIR)
    if not docs:
        print("No documents found to ingest.")
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = []
    metadatas = []
    for d in docs:
        chunks = splitter.split_text(d.page_content)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({"source": getattr(d, 'metadata', {}).get('source', 'unknown'), "page": i})
    # Create embeddings and add to collection
    collection.add(documents=texts, metadatas=metadatas, ids=[str(i) for i in range(len(texts))])
    client.persist()
    print(f"Ingested {len(texts)} chunks into ChromaDB at {PERSIST_DIR}")

if __name__ == '__main__':
    main()
