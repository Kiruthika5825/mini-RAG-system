# rag_utils.py
from langchain_milvus import Milvus
from app.services.Embeddings import embed_documents
from openai import OpenAI
import os

def init_vectorstore(collection_name="wikipedia_article"):
    """
    Initialize LangChain Milvus wrapper using your existing collection.
    """
    from pymilvus import connections
    connections.connect("default", host="localhost", port="19530")
    
    vectorstore = Milvus(
        collection_name=collection_name,
        connection_args={"host": "localhost", "port": "19530"},
        embedding_function=embed_documents
    )
    return vectorstore

def get_llm_answer(context, question):
    from openai import OpenAI
    import os

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"))

    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = client.chat.completions.create(
        model="meta/llama-3.2-3b-instruct",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def retrieve_documents(vectorstore, question, top_k=3):
    """
    Retrieve top-k relevant chunks from Milvus for the given query.
    """
    # 1. Embed the user query
    query_vector = embed_documents([question])[0]
    
    # 2. Retrieve top-k similar documents
    docs = vectorstore.similarity_search_by_vector(query_vector, k=top_k)
    return docs

def generate_rag_answer(vectorstore, question, top_k=3):
    """
    Full RAG function: embed query, retrieve relevant docs, and generate final answer.
    """
    # 1. Retrieve top-k relevant chunks
    docs = retrieve_documents(vectorstore, question, top_k=top_k)
    
    # 2. Combine content as context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. Generate answer using LLM
    answer = get_llm_answer(context, question)
    
    return answer, docs
