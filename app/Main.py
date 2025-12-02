# Main.py

from app.services.detector import is_url, detect_input_type
from app.services.loaders.txt_extractor import process_txt
from app.services.loaders.pdf_extractor import process_pdf
from app.services.loaders.docx_extractor import process_docx
from app.services.loaders.image_extractor import process_image
from app.services.loaders.url_extractor import process_url

from app.services.Embeddings import convert_to_langchain_docs, split_documents, embed_documents
from app.services.vector_db import init_collection, insert_documents, create_index_if_missing
from app.services.rag import init_vectorstore, generate_rag_answer, retrieve_documents
from app.services.router import route_to_loader

collection = init_collection()
vectorstore = init_vectorstore()

# ==============================
# 📌 PROCESS & STORE ANY FILE/URL
# ==============================

def process_and_store_input(input_path):
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


# ==================================================
# 🔍 QUERY RAG → Retrieve + LLM Answer Generation
# ==================================================

def ask(question, top_k=4):
    answer, docs = generate_rag_answer(vectorstore, question, top_k)

    print("\n===== 🧠 RAG ANSWER =====")
    print(answer)

    print("\n===== 🔎 Retreived Document Metadata =====")
    for d in docs:
        print(d.metadata)

    return answer


def process_local_file(path):
    data = route_to_loader(path)
    docs = convert_to_langchain_docs(data)
    chunks = split_documents(docs)
    embeddings = embed_documents(chunks)
    insert_documents(collection, chunks, embeddings)
    return len(chunks)


def similarity_search(text, k=3):
    return retrieve_documents(vectorstore, text, top_k=k)


def get_all_documents(limit=100):
    from pymilvus import Collection
    return collection.query(expr="id >= 0", output_fields=["text", "source_url", "title", "type", "chunk_index"], limit=limit)


def reset_collection():
    from pymilvus import Collection, utility
    collection_name = collection.name
    collection.release()
    utility.drop_collection(collection_name)
    print("Collection dropped! Run again to re‑init.")

# ==============================
# SYSTEM INITIALIZATION
# ==============================

if __name__ == "__main__":

    # 1️⃣ Initialize Vector DB (runs only once at startup)
    collection = init_collection(collection_name="knowledge_base_vectors", vector_dim=384)
    create_index_if_missing(collection)
    collection.load()

    # 2️⃣ Initialize Vectorstore for RAG search
    vectorstore = init_vectorstore(collection_name="knowledge_base_vectors")

    # ================ TEST: LOAD DATA =================

    # 🔹 Example 2 — Load a PDF file
    print("\n🔵 Loading Local PDF...")
    process_and_store_input(r"app/tests/sample.pdf")

    # 🔹 Example 3 — Load DOCX / TXT / IMAGE also works
    # process_and_store_input("notes.txt")
    # process_and_store_input("resume.docx")
    # process_and_store_input("document_scan.png")

    # ================ TEST: ASK QUESTIONS =================

    ask("what is the process of private app setup", top_k=4)
