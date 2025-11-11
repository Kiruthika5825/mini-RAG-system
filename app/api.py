from fastapi import FastAPI
from pydantic import BaseModel
from app.Main import process_user_url, answer_retriever

app = FastAPI()

class LoadRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/load")
def load_data(request: LoadRequest):
    process_user_url(request.url)
    return {"message": f"Loaded data from {request.url} into Milvus"}

@app.post("/query")
def query_data(request: QueryRequest):
    answer, docs = answer_retriever(request.question, top_k=request.top_k)
    sources = [doc.metadata for doc in docs]
    return {"answer": answer, "sources": sources}
