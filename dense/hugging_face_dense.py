from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dense(texts: List[str]) -> List[List[float]]:
    """
    Generate dense embeddings from a list of texts using a HuggingFace Dense Model.
    Returns a list of dense vectors (list of floats).
    """

    logger.info("Step 1: Choose a dense embedding model")
    # md = "sentence-transformers/all-MiniLM-L6-v2"
    # md = "BAAI/bge-small-en"
    # md = "BAAI/bge-base-en"
    md = "sentence-transformers/all-MiniLM-L6-v2"

    logger.info("Step 2: Load tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(md)
    model = AutoModel.from_pretrained(md)

    logger.info("Step 3: Tokenize input texts")
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    logger.info("Step 4: Forward pass through the model")
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

    logger.info("Step 5: Mean pooling to get sentence embeddings")
    attention_mask = inputs['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    sentence_embeddings = sum_embeddings / sum_mask  # [batch, hidden_size]

    dense_vectors = sentence_embeddings.cpu().tolist()
    
    # def get_dim():
    #     vec = list(model.embed(["a"]))[0]
    #     return len(vec)
    
    logger.info("DenseVector creation completed")
    return dense_vectors, model.config.hidden_size

# Example usage:
if __name__ == "__main__":
    texts = ["Qdrant is a vector database", "Sentence transformers"]
    dense_vectors, dim = get_dense(texts)
    print('dim:', dim)
    for vec in dense_vectors:
        logger.info(f"Generated DenseVector (dim={len(vec)}): {vec[:5]}...")
