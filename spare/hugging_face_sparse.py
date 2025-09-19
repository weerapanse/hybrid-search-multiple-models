from typing import List
from transformers import AutoTokenizer, AutoModelForMaskedLM
from qdrant_client.models import SparseVector
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sparse(texts: List[str]) -> SparseVector:
    """
    Generate a sparse embedding from a list of texts using a HuggingFace Sparse Model.
    Returns a Qdrant SparseVector.
    """

    logger.info("Step 1: Choose the sparse model")
    # OpenSearch Neural Sparse Models
    # md = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    # md = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"
    
    # BGE-M3 (Bidirectional Generative Encoder)
    # md = "BAAI/bge-m3"
    
    # Granite Embedding Models
    # md ="ibm-granite/granite-embedding-30m-sparse"
    
    # SPLADE Models (Sparse Lexical and Expansion Model)
    # md = "prithivida/Splade_PP_en_v2"
    # md = "naver/splade-cocondenser-selfdistil"
    md = "naver/splade-cocondenser-ensembledistil"

    logger.info("Step 2: Load tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(md)
    model = AutoModelForMaskedLM.from_pretrained(md)

    logger.info("Step 3: Tokenize input texts")
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    logger.debug(f"Tokenized inputs: {inputs}")

    logger.info("Step 4: Forward pass through the model")
    with torch.no_grad():
        outputs = model(**inputs)  # outputs.logits will be used for sparse embeddings
    logger.debug(f"Model output shape: {outputs.logits.shape}")

    logger.info("Step 5: Apply ReLU to logits to keep non-negative values")
    sparse_embeddings = torch.relu(outputs.logits)  # shape: [batch, seq_len, vocab_size]
    logger.debug(f"Sparse embeddings shape: {sparse_embeddings.shape}")

    logger.info("Step 6: Pooling across sequence dimension (max pooling)")
    sparse_vecs = torch.max(sparse_embeddings, dim=1).values  # shape: [batch, vocab_size]
    logger.debug(f"Pooled sparse vector shape: {sparse_vecs.shape}")

    logger.info("Step 7: Extract non-zero indices and corresponding values")
    
    logger.info("Step 8: Build Qdrant SparseVector")
    sparse_vectors = []
    for i in range(sparse_vecs.size(0)):
        vec = sparse_vecs[i]
        indices = torch.nonzero(vec).squeeze().tolist()
        if isinstance(indices, int):
            indices = [indices]
        values = vec[indices].tolist()
        sparse_vectors.append(SparseVector(
            indices=[str(idx) for idx in indices],
            values=values
        ))

    logger.info("SparseVector creation completed")
    return sparse_vectors

# Example usage:
if __name__ == "__main__":
    texts = ["Qdrant is a vector database", "Sentence transformers"]
    sparse_vectors = get_sparse(texts)
    for vec in sparse_vectors:
        logger.info(f"Generated SparseVector: {vec}")
