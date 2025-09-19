import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import models
from sentence_transformers import CrossEncoder
from collections import OrderedDict
from dense.hugging_face_dense import get_dense
from spare.hugging_face_sparse import get_sparse
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
QDRANT_URL= os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLLECTION_NAME = 'document_384'

def get_rerank(questions: List[str]):
    dense, dim  = get_dense(questions)
    sparse = get_sparse(questions)
    results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense[0],
                    using="dense",
                    limit=20,
                ),
                models.Prefetch(
                    query=sparse[0],
                    using="sparse",
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=10,
    )

    unique_candidates = OrderedDict()
    for p in results.points:
        text = p.payload["text"]
        if text not in unique_candidates:
            unique_candidates[text] = p.score
        else:
            unique_candidates[text] = max(unique_candidates[text], p.score)
    candidates = list(unique_candidates.keys())
    
    reranker = CrossEncoder("NeginShams/cross_encoder_v2", trust_remote_code=True)
    pairs = [(questions[0], doc) for doc in candidates]
    scores = reranker.predict(pairs)
    scores = np.array(scores)
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    final_ranked = sorted(zip(candidates, scores_norm), key=lambda x: x[1], reverse=True)
    return final_ranked

if __name__ == "__main__":
    questions = ["Qdrant"]
    ranked = get_rerank(questions)
    context = "\n".join([doc for doc, _ in ranked])
    print(context)
    for doc, score in ranked:
        print(f"{score:.4f} | {doc}")