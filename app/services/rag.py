# rag_utils.py (UPDATED & CLEAN)

import os
from openai import OpenAI
from langchain_milvus import Milvus
from app.services.Embeddings import embed_documents
from langchain_community.embeddings import HuggingFaceEmbeddings

def init_vectorstore(collection_name="knowledge_base_vectors", host="localhost", port="19530"):

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Milvus(
        embedding_function=embedding_function,   # ← REQUIRED
        collection_name=collection_name,
        connection_args={"host": host, "port": port},
        search_params={"metric_type": "COSINE", "params": {"ef": 64}}
    )

    print(f"Milvus VectorStore initialized → {collection_name}")
    return vectorstore


def retrieve_documents(vectorstore, query, top_k=5):
    """
    Embeds user query → retrieves similar chunks.
    """
    query_emb = embed_documents([query])[0]

    results = vectorstore.similarity_search_by_vector(
        embedding=query_emb,
        k=top_k,
        expr=None  # optional filters later like type == 'pdf'
    )

    return results



def get_llm_answer(context, question):
    """
    LLM call — now clean & reusable.
    """

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )

    prompt = f"""
You are a RAG assistant. Answer based ONLY on the given context.
If information is missing, respond with "Not enough data found in memory."

Context:
{context}

Question: {question}
Answer:
"""

    response = client.chat.completions.create(
        model="meta/llama-3.2-3b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )

    return response.choices[0].message.content.strip()



def generate_rag_answer(vectorstore, question, top_k=5):
    """
    Complete RAG cycle:
    - embed query → retrieve docs → build context → get answer
    """

    docs = retrieve_documents(vectorstore, question, top_k=top_k)

    if not docs:
        return "No relevant memory found.", []

    context = "\n\n".join([doc.page_content for doc in docs])
    answer = get_llm_answer(context, question)

    return answer, docs
