from pymilvus import (connections, utility, FieldSchema, CollectionSchema, DataType, Collection)

def init_collection(collection_name="wikipedia_article", vector_dim=384):
    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")

    # Check if collection already exists
    if collection_name in utility.list_collections():
        collection = Collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
        return collection

    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=300),
        FieldSchema(name="chunk_index", dtype=DataType.INT64)
    ]

    schema = CollectionSchema(fields, description="Wikipedia Articles Collection")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created.")
    return collection

def insert_documents(collection, split_docs, embeddings):

    if not split_docs or len(embeddings) == 0:
        print("No documents or embeddings to insert.")
        return

    texts = [d.page_content for d in split_docs]
    urls = [d.metadata["source_url"] for d in split_docs]
    titles = [d.metadata["title"] for d in split_docs]
    chunk_indices = [d.metadata["chunk_index"] for d in split_docs]

    # Insert into Milvus
    collection.insert([texts, embeddings, urls, titles, chunk_indices])
    
    # Flush to make data persistent and visible in Attu
    collection.flush()
    print(f"Inserted and flushed {len(split_docs)} chunks into '{collection.name}'")

    # Check how many entities are in the collection
    print("Number of entities in collection:", collection.num_entities)


# creating index
def create_index_if_missing(collection, field_name="vector"):
    """
    Create an index on the specified vector field if it doesn't exist.
    """
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 8, "efConstruction": 64}
    }

    # Check if an index already exists for this field
    existing_indexes = [idx for idx in collection.indexes if idx.field_name == field_name]
    if not existing_indexes:
        collection.create_index(field_name=field_name, index_params=index_params)
        print(f"Index created on collection '{collection.name}' for field '{field_name}'")
    else:
        print(f"Collection '{collection.name}' already has an index on field '{field_name}'")
