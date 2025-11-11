#Embedding
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def convert_to_langchain_docs(data):
    docs = [
        Document(
            page_content=item["text"],
            metadata={
                "source_url": item["source_url"],
                "title": item["title"],
                "chunk_index": item["chunk_index"]
            }
        )
        for item in data
    ]
    return docs


def split_documents(docs, chunk_size=600, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = []
    for doc in docs:
        chunks = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            split_docs.append(
                Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk_index": i}
                )
            )
    return split_docs


# Load environment variables from .env file
load_dotenv()

def embed_documents(docs, model_name=None):
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)

    # handle both strings and Documents
    if isinstance(docs[0], str):
        texts = docs
    elif isinstance(docs[0], Document):
        texts = [doc.page_content for doc in docs]
    else:
        raise ValueError("embed_documents expects a list of Documents or strings")

    embeddings = model.encode(texts)
    return embeddings