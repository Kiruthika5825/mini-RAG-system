import os
from openai import OpenAI
from langchain_milvus import Milvus
from app.services.Embeddings import embed_documents
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util
import asyncio


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



async def generate_rag_answer(vectorstore, question, top_k=5):
    """
    Full RAG cycle:
    - retrieve docs
    - create answer
    - evaluate answer
    """

    docs = retrieve_documents(vectorstore, question, top_k=top_k)

    if not docs:
        return {
            "answer": "No relevant memory found.",
            "rag_score": 0,
            "details": {},
            "docs": []
        }

    context = "\n\n".join([doc.page_content for doc in docs])
    answer = get_llm_answer(context, question)

    eval_scores = await evaluate_rag(question, docs, answer)

    return {
        "answer": answer,
        "rag_score": eval_scores["rag_score"],
        "details": eval_scores,
        "docs": docs
    }


model_eval = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

async def evaluate_rag(query, retrieved_docs, answer):
    """
    Evaluates RAG answer quality:
    - Context Recall (similarity: query ↔ retrieved)
    - Faithfulness (answer supported by retrieved context)
    - Final RAG Score (percentage)
    """

    # --- 1. Context Recall ---
    query_emb = model_eval.encode(query, convert_to_tensor=True)
    doc_embs = model_eval.encode(
        [doc.page_content for doc in retrieved_docs],
        convert_to_tensor=True
    )

    similarities = util.cos_sim(query_emb, doc_embs)[0]
    context_recall = float(similarities.max().item())

    # --- 2. Faithfulness ---
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )

    context_text = "\n\n".join([d.page_content for d in retrieved_docs])

    check_prompt = f"""
Evaluate if the ANSWER is fully supported by the CONTEXT.
Return ONLY a floating number between 0 and 1.

CONTEXT:
{context_text}

ANSWER:
{answer}
"""

    faith_resp = client.chat.completions.create(
        model="meta/llama-3.2-3b-instruct",
        messages=[{"role": "user", "content": check_prompt}],
        max_tokens=10
    )

    try:
        faithfulness = float(faith_resp.choices[0].message.content.strip())
    except:
        faithfulness = 0.0

    rag_score = (0.6 * context_recall + 0.4 * faithfulness) * 100

    return {
        "context_recall": round(context_recall, 3),
        "faithfulness": round(faithfulness, 3),
        "rag_score": round(rag_score, 2)
    }

def generate_rag_answer_with_eval(vectorstore, question, top_k=4):
    result = asyncio.run(generate_rag_answer(vectorstore, question, top_k))

    return {
        "answer": result["answer"],
        "score": result["rag_score"],
        "docs": result["docs"]
    }
