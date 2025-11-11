from app.services.scraper import get_paragraphs
from app.services.Embeddings import convert_to_langchain_docs, split_documents, embed_documents
from app.services.vector_db import init_collection, insert_documents, create_index_if_missing
from app.services.rag import init_vectorstore, generate_rag_answer, retrieve_documents  # import RAG functions

# Initialize Milvus collection
collection = init_collection(vector_dim=384)
create_index_if_missing(collection, field_name="vector")
collection.load()

def process_user_url(url: str):
    # 1️ Scrape page
    data = get_paragraphs(url)
    print(f"Fetched {len(data)} paragraphs from {url}")

    # 2️ Convert to LangChain Documents
    docs = convert_to_langchain_docs(data)
    print(f"Converted to {len(docs)} documents")

    # 3️ Split documents into chunks
    split_docs = split_documents(docs, chunk_size=600, chunk_overlap=100)
    print(f"Split into {len(split_docs)} chunks")

    # 4️ Generate embeddings
    embeddings = embed_documents(split_docs)
    print(f"Generated embeddings for {len(embeddings)} chunks")
    print("Sample embedding length:", len(embeddings[0]))

    # 5️ Insert into Milvus
    insert_documents(collection, split_docs, embeddings)
    create_index_if_missing(collection, field_name="vector")
    collection.load()



# Initialize LangChain Milvus wrapper once at startup
vectorstore = init_vectorstore()

def answer_retriever(question: str, top_k: int = 3):
    """
    Retrieve relevant chunks from Milvus and generate answer using RAG.
    """
    answer, docs = generate_rag_answer(vectorstore, question, top_k=top_k)
    return answer, docs



# testing
if __name__ == "__main__":
    # Load content from URL
    user_url = "https://en.wikipedia.org/wiki/Data_science"
    process_user_url(user_url)

    # Query RAG
    user_question = "What are the main responsibilities of a data scientist?"
    answer, source_docs = answer_retriever(user_question, top_k=4)

    print("\n=== Answer ===")
    print(answer)

    print("\n=== Source Chunks Metadata ===")
    for doc in source_docs:
        print(doc.metadata)
