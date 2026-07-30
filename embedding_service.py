"""
Embedding Service
=================
This module is responsible for generating vector embeddings for text chunks.
It uses the SentenceTransformers library and the BAAI/bge-small-en-v1.5 model.
The module is entirely independent of storage (e.g., Qdrant) or ingestion (e.g., OCR) logic.
"""

import logging
from typing import List, Dict, Any

# We use SentenceTransformer to load the model and compute embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

# Configure logging for the embedding module
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model Configuration
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Load the model once at module initialization to avoid repeated loading overhead
logger.info(f"Loading embedding model: {MODEL_NAME}")
try:
    _model = SentenceTransformer(MODEL_NAME)
    logger.info("Embedding model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load the embedding model '{MODEL_NAME}': {e}")
    raise


def embed_text(text: str) -> List[float]:
    """
    Generates an embedding for a single text string.

    Args:
        text (str): The text to embed.

    Returns:
        List[float]: A list of floats representing the embedding vector.
                     Returns an empty list if the text is empty or an error occurs.
    """
    if not text or not text.strip():
        logger.warning("Received empty text for embedding. Skipping.")
        return []

    try:
        # Generate the embedding. Output is a numpy array.
        # We convert it to a flat Python list of floats as required.
        # normalize_embeddings=True is highly recommended for BGE models (cosine similarity)
        # Add BGE document prefix 'passage: ' to the text internally before encoding
        embedding = _model.encode(
            f"passage: {text}", 
            normalize_embeddings=True,
            batch_size=32
        )
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding for text: {e}")
        return []


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes a list of chunk dictionaries, generates embeddings for their 'text' content,
    and appends an 'embedding' field to each dictionary.

    Args:
        chunks (List[Dict[str, Any]]): A list of chunk metadata dictionaries.
                                       Each dictionary must contain a 'text' key.

    Returns:
        List[Dict[str, Any]]: The original list of dictionaries, but with an 'embedding'
                              key added to chunks that have valid text.
    """
    if not chunks:
        logger.warning("Received empty list of chunks.")
        return []

    logger.info(f"Processing {len(chunks)} chunks for embeddings...")
    
    # Optimize by batch-encoding all texts at once rather than in a loop.
    # First, collect all valid texts and their indices (to safely skip empty chunks)
    texts_to_embed = []
    valid_indices = []

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        if text and text.strip():
            # Add 'passage: ' prefix as per BGE best practices, used only internally during embedding
            texts_to_embed.append(f"passage: {text}")
            valid_indices.append(i)
        else:
            logger.debug(f"Chunk at index {i} has no text. Skipping embedding.")

    if not texts_to_embed:
        logger.warning("No valid text found in chunks to embed.")
        return chunks

    try:
        # Encode all collected texts in a batch
        # Added batch_size=32 and show_progress_bar=True for better performance and UX on large documents
        embeddings = _model.encode(
            texts_to_embed, 
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True
        )
        
        # Map the embeddings back to the original chunks using their indices
        for idx, emb in zip(valid_indices, embeddings):
            chunks[idx]["embedding"] = emb.tolist()
            
        logger.info(f"Successfully added embeddings to {len(texts_to_embed)} chunks.")
    except Exception as e:
        logger.error(f"Error during batch embedding generation: {e}")

    # Return the chunks in their original order, safely mutated.
    return chunks
