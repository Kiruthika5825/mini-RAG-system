from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
from contextlib import asynccontextmanager

from app.Main import (
    process_and_store_input,
    process_local_file
)

# Globals set during startup
vectorstore = None
collection = None


# =====================================================
# 🔄 Lifespan (replaces deprecated @app.on_event)
# =====================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, collection

    print("🔄 Initializing Milvus collection & vectorstore...")

    from app.services.vector_db import init_collection, create_index_if_missing
    from app.services.rag import init_vectorstore

    # Initialize Milvus collection
    collection = init_collection(collection_name="knowledge_base_vectors", vector_dim=384)

    # Create index if missing
    create_index_if_missing(collection)

    # Load into memory for search
    collection.load()

    # Initialize vectorstore for RAG
    vectorstore = init_vectorstore(collection_name="knowledge_base_vectors")

    print("✅ Vectorstore Ready")

    yield  # App runs here

    print("🔻 Shutting down...")


# Initialize app with lifespan
app = FastAPI(title="RAG API", version="1.1", lifespan=lifespan)


# ===================================================
# REQUEST MODELS
# ===================================================

class LoadURL(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class SearchRequest(BaseModel):
    text: str
    top_k: int = 3


# ===================================================
# LOAD / INGEST ENDPOINTS
# ===================================================

@app.post("/load/url")
def load_from_url(req: LoadURL):
    """Extract → embed → store from URL"""
    global collection
    chunks_indexed = process_and_store_input(req.url, collection)
    return {"status": "success", "message": f"URL indexed: {req.url}", "chunks_indexed": chunks_indexed}


@app.post("/load/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload PDF/TXT/DOCX/IMAGE from UI"""
    global collection
    os.makedirs("uploads", exist_ok=True)
    save_path = f"uploads/{file.filename}"

    with open(save_path, "wb") as f:
        f.write(await file.read())

    chunks = process_local_file(save_path, collection)

    return {"status": "success", "file": file.filename, "chunks_indexed": chunks}


# ===================================================
# QUERY / RAG ENDPOINTS
# ===================================================

@app.post("/query")
def rag_query(req: QueryRequest):
    global vectorstore
    from app.services.rag import generate_rag_answer_with_eval

    result = generate_rag_answer_with_eval(vectorstore, req.question, req.top_k)

    return {
        "answer": result["answer"],
        "evaluation_score": result["score"],
        "retrieved_documents": result["docs"]
    }
