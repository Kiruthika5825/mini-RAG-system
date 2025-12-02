# vector_db.py (FULLY UPDATED)

from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema,
    DataType, Collection
)

# 🔹 Default DB Name Updated
DEFAULT_DB_NAME = "knowledge_base_vectors"
VECTOR_DIM = 384   # must match embedding model


def connect_to_milvus(host="localhost", port="19530"):
    connections.connect("default", host=host, port=port)
    print(f"Connected to Milvus → {host}:{port}")


# ---------------------------------------------------------
# 1️⃣ Create / Load Collection with full schema support
# ---------------------------------------------------------
def init_collection(collection_name=DEFAULT_DB_NAME, vector_dim=VECTOR_DIM):
    connect_to_milvus()

    # ─ If exists → load instead of creating new one
    if collection_name in utility.list_collections():
        print(f"⚠ Collection '{collection_name}' already exists. Loading it...")
        return Collection(collection_name)
    else:
        print(f"🆕 Creating collection '{collection_name}' with COSINE metric...")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),

        # --- Data Fields ---
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=50),   # NEW
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=300),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),

        # --- Vector Field ---
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    ]

    schema = CollectionSchema(fields, description="Multi‑format knowledge embeddings")

    collection = Collection(name=collection_name, schema=schema)
    print("🔨 Creating HNSW (COSINE) index...")
    collection.create_index(
            "vector",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 12, "efConstruction": 100}
            }
        )
    print("✅ COSINE index ready!")

    return collection
# ---------------------------------------------------------
# 2️⃣ Insert Documents into Milvus
# ---------------------------------------------------------
def insert_documents(collection, split_docs, embeddings):
    if not split_docs or len(split_docs) != len(embeddings):
        print("❌ Error: Docs & embeddings count mismatch!")
        return

    print(f"📥 Inserting {len(split_docs)} chunks into Milvus...")

    texts          = [d.page_content       for d in split_docs]
    types          = [d.metadata.get("type", "")   for d in split_docs]
    titles         = [d.metadata.get("title", "")  for d in split_docs]
    sources        = [d.metadata.get("source","")  for d in split_docs]
    chunk_indices  = [d.metadata["chunk_index"]    for d in split_docs]
    vectors        = embeddings

    collection.insert([
        texts,
        types,
        titles,
        sources,
        chunk_indices,
        vectors
    ])

    collection.flush()
    print(f"🟩 Successfully inserted {len(split_docs)} vectors.")
    print("Total stored:", collection.num_entities)


# ---------------------------------------------------------
# 3️⃣ Create HNSW Index (Only if Missing)
# ---------------------------------------------------------
def create_index_if_missing(collection, field_name="vector"):
    found = any(idx.field_name == field_name for idx in collection.indexes)

    if found:
        print(f"Index already exists on '{field_name}' — OK")
        return

    print("⚙ Creating HNSW index...")

    collection.create_index(
        field_name=field_name,
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200}
        }
    )

    print("🧩 Index Created Successfully!")


# ---------------------------------------------------------
# Quick Constructor Helper for Routers
# ---------------------------------------------------------
def setup_vector_db(name=DEFAULT_DB_NAME):
    collection = init_collection(name)
    create_index_if_missing(collection)
    return collection
