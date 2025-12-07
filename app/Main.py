# Main.py

from app.services.detector import detect_input_type
from app.services.loaders.txt_extractor import process_txt
from app.services.loaders.pdf_extractor import process_pdf
from app.services.loaders.docx_extractor import process_docx
from app.services.loaders.image_extractor import process_image
from app.services.loaders.url_extractor import process_url

from app.services.Embeddings import convert_to_langchain_docs, split_documents, embed_documents
from app.services.vector_db import insert_documents
from app.services.rag import retrieve_documents, generate_rag_answer_with_eval
from app.services.router import route_to_loader


# ==============================
# 📌 PROCESS & STORE ANY FILE/URL
# ==============================
def process_and_store_input(input_path, collection):
    """Extracts text from any supported input → chunks → embeds → stores in Milvus."""
    input_type = detect_input_type(input_path)

    if input_type == "url":
        raw_data = process_url(input_path)
    elif input_type == "txt":
        raw_data = process_txt(input_path)
    elif input_type == "pdf":
        raw_data = process_pdf(input_path)
    elif input_type == "docx":
        raw_data = process_docx(input_path)
    elif input_type in ["png", "jpg", "jpeg"]:
        raw_data = process_image(input_path)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    print(f"\n📄 Extracted {len(raw_data)} content chunks from → {input_path}")

    # Convert → LangChain documents
    docs = convert_to_langchain_docs(raw_data)

    # Chunk the data
    chunks = split_documents(docs, chunk_size=600, chunk_overlap=100)
    print(f"🔹 After chunking → {len(chunks)} chunks")

    # Generate embeddings
    embeddings = embed_documents(chunks)
    print("🔹 Embeddings Created")

    # Store in DB
    insert_documents(collection, chunks, embeddings)
    collection.load()
    print("\n💾 Stored Successfully in Vector DB\n")
    return len(chunks)


def process_local_file(path, collection):
    """Process a local file (PDF, TXT, DOCX, IMAGE) → embed → store in collection."""
    data = route_to_loader(path)
    docs = convert_to_langchain_docs(data)
    chunks = split_documents(docs)
    embeddings = embed_documents(chunks)
    insert_documents(collection, chunks, embeddings)
    return len(chunks)


# ==================================================
# 🔍 QUERY RAG → Retrieve + LLM Answer Generation
# ==================================================
def ask(vectorstore, question, top_k=4):
    """Retrieve docs & generate LLM answer using RAG."""
    result = generate_rag_answer_with_eval(vectorstore, question, top_k)
    
    print("\n===== 🧠 RAG ANSWER =====")
    print(result["answer"])

    print("\n===== 🔎 Retrieved Document Metadata =====")
    for d in result["docs"]:
        print(d.metadata)

    return result["answer"]


def similarity_search(vectorstore, text, k=3):
    """Retrieve top-k similar documents from the vectorstore."""
    return retrieve_documents(vectorstore, text, top_k=k)


def get_all_documents(collection, limit=100):
    """Query all stored documents (up to limit)."""
    return collection.query(
        expr="id >= 0",
        output_fields=["text", "source", "title", "type", "chunk_index"],
        limit=limit
    )


def reset_collection(collection):
    """Drop the Milvus collection completely."""
    from pymilvus import utility

    collection_name = collection.name
    collection.release()
    utility.drop_collection(collection_name)
    print("Collection dropped! Run again to re‑init.")
