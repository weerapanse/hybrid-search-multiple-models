from typing import List
from fastembed import TextEmbedding
from qdrant_client.models import VectorParams
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dense(texts: List[str]) -> List[List[float]]:
    """
    Generate dense embeddings from a list of texts using FastEmbed.
    Returns a list of dense vectors (list of floats).
    """
    logger.info("Step 1: Initialize FastEmbed dense model")
    md = "sentence-transformers/all-MiniLM-L6-v2"
    model = TextEmbedding(model_name=md)

    logger.info("Step 2: Embed texts")
    dense_vectors = list(model.embed(texts))
    logger.info(f"Number of dense embeddings generated: {len(dense_vectors)}")

    for i, dvec in enumerate(dense_vectors):
        logger.info(f"Step 3: Processing text {i}: {texts[i]}")
        logger.debug(f"dense[:10]: {dvec[:10]}")  # Show first 10 dims
    
    def get_dim():
        vec = list(model.embed(["a"]))[0]
        return len(vec)
    
    logger.info("All dense vectors created")
    return dense_vectors, get_dim()


# Example usage:
if __name__ == "__main__":
    texts = ["Qdrant is a vector database", "Sentence transformers"]
    dense_vectors, dim = get_dense(texts)
    print('dim:', dim)
    for vec in dense_vectors:
        logger.info(f"Generated DenseVector (dim={len(vec)}): {vec[:5]}...")
