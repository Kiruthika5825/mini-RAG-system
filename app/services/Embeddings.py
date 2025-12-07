# Embeddings.py

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()


# Convert raw loader output → LangChain Document objects
def convert_to_langchain_docs(data):
    docs = [
        Document(
            page_content=item["text"],
            metadata={
                "source": item["source"], 
                "title": item.get("title", ""),
                "type": item.get("type", ""),
                "chunk_index": item["chunk_index"]
            }
        )
        for item in data
    ]
    return docs
 

# Split long documents into smaller chunks
def split_documents(docs, chunk_size=600, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = []
    for doc in docs:
        chunks = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):   # reset chunk index for split chunks
            split_docs.append(
                Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk_index": i}
                )
            )
    return split_docs


# Embed chunks using SentenceTransformer
def embed_documents(docs, model_name=None):
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    model = SentenceTransformer(model_name)

    # Accept both list[str] and list[Document]
    if isinstance(docs[0], str):
        texts = docs
    elif isinstance(docs[0], Document):
        texts = [doc.page_content for doc in docs]
    else:
        raise ValueError("embed_documents expects a list of Documents or strings")

    embeddings = model.encode(texts)
    return embeddings
