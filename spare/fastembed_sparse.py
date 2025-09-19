from typing import List
from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sparse(texts: List[str]) -> List[SparseVector]:
    """
    Generate sparse embeddings from a list of texts using FastEmbed.
    Returns a list of Qdrant SparseVectors.
    """
    logger.info("Step 1: Initialize FastEmbed sparse model")
    md = "prithivida/Splade_PP_en_v1"
    # md = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    # md = "Qdrant/bm25"
    # md = "Qdrant/minicoil-v1"
    model = SparseTextEmbedding(md)

    logger.info("Step 2: Embed texts")
    sparse_vecs = list(model.embed(texts))
    logger.info(f"Number of sparse embeddings generated: {len(sparse_vecs)}")

    sparse_vectors = []
    for i, svec in enumerate(sparse_vecs):
        logger.info(f"Step 3: Processing text {i}: {texts[i]}")
        indices = [str(idx) for idx in svec.indices.tolist()]
        values = svec.values.tolist()
        logger.debug(f"indices[:10]: {indices[:10]}")  # Show first 10
        logger.debug(f"values[:10]: {values[:10]}")  # Show first 10

        qdrant_sparse = SparseVector(indices=indices, values=values)
        sparse_vectors.append(qdrant_sparse)

    logger.info("All SparseVectors created")
    
    return sparse_vectors

# Example usage:
if __name__ == "__main__":
    texts = ["Qdrant is a vector database", "Sentence transformers"]
    sparse_vectors = get_sparse(texts)
    for vec in sparse_vectors:
        logger.info(f"Generated SparseVector: {vec}")
