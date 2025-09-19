import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
import os
QDRANT_URL= os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
collection_name = ''

from dense.hugging_face_dense import get_dense
from spare.hugging_face_sparse import get_sparse

def get_point(dense:List, sparse: List, docs: List[str]):
    points = []
    for i, doc in enumerate(docs):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                payload={
                    "text": doc
                },
                vector={
                    "dense": dense[i],
                    "sparse": sparse[i]
                }
            )
        )
    return points

def upsert(collection_name: str, points: List):
    client.upsert(collection_name=collection_name, points=points)
    print("Upsert points successfully")

def create_collection(dim: int):
    collection_name = f'document_{dim}'
    distance="Cosine"
    client.recreate_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=dim, distance=distance)
                },
                sparse_vectors_config={
                    "sparse": {}
                }
    )
    return collection_name, dim, distance

# Example usage:
if __name__ == "__main__":
    docs = [
        "Qdrant is a vector database for the next generation of AI applications.",
        "FastEmbed provides state-of-the-art embedding models.",
        "Sentence Transformers are widely used for dense retrieval.",
        "Splade is a sparse model that expands queries for better recall."
    ]
    dense, dim = get_dense(docs)
    sparse = get_sparse(docs)
    points = get_point(dense=dense, sparse=sparse, docs=docs)
    collections = client.get_collections()
    coll_match = False
    for col in collections.collections:
        coll_name = col.name
        info = client.get_collection(col.name)
        vectors_config = info.config.params.vectors
        if vectors_config is not None and "dense" in vectors_config:
            if int(dim) == int(vectors_config["dense"].size) and 'Cosine' == vectors_config["dense"].distance:
                print(f'Found collection name: {coll_name} dim: {dim}')
                collection_name = coll_name
                coll_match = True
    if not coll_match:
        coll_name, dim, distance = create_collection(dim=dim)
        collection_name = coll_name
        print(f'Created collection name: {coll_name} dim: {dim} distance: {distance}')
            
    upsert(collection_name=collection_name, points=points)