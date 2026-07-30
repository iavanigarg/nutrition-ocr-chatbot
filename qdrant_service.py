import logging
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, UpdateStatus, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "nutrition_chunks"
VECTOR_SIZE = 384
DISTANCE_METRIC = Distance.COSINE

# Initialize Qdrant Client
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def create_collection() -> None:
    """
    Creates the Qdrant collection if it does not already exist.
    Reuses it if it already exists.
    """
    try:
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            logger.info(f"Collection '{COLLECTION_NAME}' does not exist. Creating it...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE_METRIC),
                on_disk_payload=True
            )
            logger.info(f"Collection '{COLLECTION_NAME}' created successfully.")
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists. Reusing it.")
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise

def _generate_point_id(chunk_id: str) -> str:
    """
    Generates a deterministic UUID from the chunk_id.
    Qdrant requires point IDs to be integers or UUIDs. This ensures
    duplicate uploads overwrite the same point instead of creating duplicates.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

def store_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Batch uploads document chunks to the Qdrant collection.
    
    Args:
        chunks: A list of dictionaries representing chunks. Each must contain an
                'embedding' key and other metadata fields (like 'chunk_id').
    """
    if not chunks:
        logger.warning("No chunks provided for storage.")
        return

    logger.info(f"Received {len(chunks)} chunks for processing.")

    points = []
    skipped_chunks = 0

    for chunk in chunks:
        try:
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                logger.warning(f"Skipping chunk missing 'chunk_id': {chunk}")
                skipped_chunks += 1
                continue
            
            if "embedding" not in chunk:
                logger.warning(f"Chunk '{chunk_id}' is missing required field 'embedding'. Skipping.")
                skipped_chunks += 1
                continue

            # Keep the original chunk unchanged and extract the vector
            vector = chunk["embedding"]
            
            # Validate embedding size before uploading
            if len(vector) != VECTOR_SIZE:
                logger.warning(f"Chunk '{chunk_id}' has invalid embedding size: {len(vector)} (expected {VECTOR_SIZE}). Skipping.")
                skipped_chunks += 1
                continue
            
            # Create a payload copy excluding only the embedding field
            payload = {k: v for k, v in chunk.items() if k != "embedding"}
            
            # Generate deterministic point ID
            point_id = _generate_point_id(chunk_id)
            
            # Construct the point
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            points.append(point)
        except Exception as e:
            logger.error(f"Unexpected error processing chunk: {e}")
            skipped_chunks += 1
            
    if not points:
        logger.warning(f"No valid points were constructed. Skipped {skipped_chunks} chunks.")
        return

    try:
        logger.info(f"Batch uploading {len(points)} vectors to '{COLLECTION_NAME}'...")
        # upsert inserts new vectors or updates existing ones with the same ID
        operation_info = client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        if operation_info.status == UpdateStatus.COMPLETED:
            logger.info(f"Successfully uploaded {len(points)} vectors. Skipped: {skipped_chunks}.")
            logger.info("Upload completed.")
        else:
            logger.warning(f"Upload operation status: {operation_info.status}")
    except Exception as e:
        logger.error(f"Error uploading vectors to Qdrant: {e}")
        raise

def delete_document(document_id: str) -> None:
    """
    Deletes all vectors (chunks) associated with a specific document_id.
    
    Args:
        document_id: The ID of the document to delete.
    """
    try:
        logger.info(f"Deleting vectors for document_id: '{document_id}'...")
        operation_info = client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )
        if operation_info.status == UpdateStatus.COMPLETED:
            logger.info(f"Successfully deleted vectors for document_id: '{document_id}'.")
        else:
            logger.warning(f"Delete operation status: {operation_info.status}")
    except Exception as e:
        logger.error(f"Error deleting vectors for document_id '{document_id}': {e}")
        raise

def count_vectors() -> int:
    """
    Counts the total number of vectors in the current collection.
    
    Returns:
        The integer count of vectors.
    """
    try:
        count_result = client.count(
            collection_name=COLLECTION_NAME,
            exact=True
        )
        logger.info(f"Total vectors in '{COLLECTION_NAME}': {count_result.count}")
        return count_result.count
    except UnexpectedResponse as e:
        if e.status_code == 404:
            logger.warning(f"Collection '{COLLECTION_NAME}' not found. Returning count 0.")
            return 0
        raise
    except Exception as e:
        logger.error(f"Error counting vectors: {e}")
        raise
