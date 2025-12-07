from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os

from app.Main import (
    process_and_store_input, process_local_file, ask,
    get_all_documents, similarity_search, reset_collection
)

app = FastAPI(title="Universal RAG API", version="0.0.1")


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
    process_and_store_input(req.url)
    return {"status": "success", "message": f"URL indexed: {req.url}"}


@app.post("/load/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload PDF/TXT/DOCX/IMAGE from UI"""
    os.makedirs("uploads", exist_ok=True)
    save_path = f"uploads/{file.filename}"

    with open(save_path, "wb") as f:
        f.write(await file.read())

    chunks = process_local_file(save_path)

    return {"status": "success", "file": file.filename, "chunks_indexed": chunks}


# ===================================================
# QUERY / RAG ENDPOINTS
# ===================================================

@app.post("/query")
def rag_query(req: QueryRequest):
    """RAG Retrieval + LLM Answer"""
    answer = ask(req.question, req.top_k)
    return {"answer": answer}