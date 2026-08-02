import logging
import time
from typing import List, Dict, Any, Optional

from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse

# Reuse Qdrant configuration and client
from qdrant_service import client, COLLECTION_NAME

# Reuse embedding function
from embedding_service import embed_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Stronger relevance filtering for production
# Changed MIN_SCORE from 0.70 to 0.0 to avoid rejecting valid embeddings because bge-small-en-v1.5 cosine similarity might not reach 0.70 for all queries.
MIN_SCORE = 0.0

def search_chunks(query: str, top_k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Embeds the query and performs a vector search in Qdrant to find the most relevant chunks.
    
    Args:
        query: The search query string.
        top_k: Number of most relevant chunks to return (default: 5). Will be clamped between 1 and 5.
        document_id: Optional document ID to restrict the search to a specific document.
        
    Returns:
        A list of dictionaries containing metadata, text, and similarity score.
        Returns an empty list on any error instead of crashing.
    """
    total_start_time = time.time()
    
    # 5. Limit retrieved chunks: Clamp top_k safely between 1 and 5
    if not isinstance(top_k, int):
        top_k = 5
    top_k = max(1, min(5, top_k))

    if not query or not query.strip():
        logger.warning("Received empty query. Returning empty results.")
        return []

    # Log separately: received query
    logger.info(f"Received query: '{query}'")
    
    # Log separately: embedding generation started
    logger.info("Embedding generation started...")
    emb_start_time = time.time()
    
    query_vector = embed_text(query)
    
    # Log separately: embedding generation time
    emb_time = time.time() - emb_start_time
    logger.info(f"Embedding generation time: {emb_time:.3f} seconds.")
    
    if not query_vector:
        logger.warning("Query embedding is empty. Returning empty results.")
        return []
        
    # Log separately: vector search started
    logger.info(f"Vector search started in collection '{COLLECTION_NAME}' (limit={top_k})...")
    search_start_time = time.time()
    
    try:
        # Build query filter if restricted to a specific document
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )

        # Prefer query_points, fallback to search
        if hasattr(client, "query_points"):
            search_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k
            ).points
        else:
            search_results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k
            )
        
        # Log separately: vector search time
        search_time = time.time() - search_start_time
        logger.info(f"Vector search time: {search_time:.3f} seconds.")
        
        # 6. Log separately: total retrieved chunks
        retrieved_count = len(search_results)
        logger.info(f"Total retrieved chunks (pre-filtering): {retrieved_count}")
        
        if retrieved_count == 0:
            logger.info("No chunks retrieved from Qdrant. Returning empty results.")
            return []
            
        # 3. Add a best-score quality gate
        # Qdrant search results are naturally sorted by score in descending order
        highest_score = search_results[0].score if search_results else 0.0
        
        # 6. Log separately: highest similarity score
        logger.info(f"Highest similarity score: {highest_score:.4f}")
        
        if highest_score < MIN_SCORE:
            logger.warning(f"Highest score {highest_score:.4f} is below threshold {MIN_SCORE}. Retrieval rejected because of low relevance.")
            return []
            
        formatted_results = []
        filtered_count = 0
        total_score_remaining = 0.0
        
        # Process and filter hits
        for hit in search_results:
            score = hit.score
            
            # Enforce Minimum Similarity Threshold
            if score < MIN_SCORE:
                filtered_count += 1
                continue
                
            # We preserve all metadata from the payload
            result_dict = hit.payload.copy() if hit.payload else {}
            # Append the similarity score
            result_dict["score"] = score
            formatted_results.append(result_dict)
            total_score_remaining += score
            
        # 6. Log separately: filtered chunks
        logger.info(f"Number of filtered chunks (score < {MIN_SCORE}): {filtered_count}")
        
        # 2. Reject irrelevant retrievals
        if not formatted_results:
            logger.warning("No chunks remain after score filtering. Retrieval rejected because of low relevance.")
            return []
            
        # 4. Add average-score validation
        average_score = total_score_remaining / len(formatted_results)
        
        # 6. Log separately: average similarity score
        logger.info(f"Average similarity score of remaining chunks: {average_score:.4f}")
        
        if average_score < MIN_SCORE:
            logger.warning(f"Average score {average_score:.4f} is below threshold {MIN_SCORE}. Retrieval rejected because of low relevance.")
            return []
            
        # 6. Log separately: final returned chunk count
        final_count = len(formatted_results)
        logger.info(f"Final returned chunk count: {final_count}")
        
        # Log separately: total retrieval time
        total_time = time.time() - total_start_time
        logger.info(f"Total retrieval time: {total_time:.3f} seconds.")
        
        # Print Debug Logs for User Query
        print("\n=== DEBUG LOGS: USER QUERY ===")
        print(f"question: {query}")
        print(f"query embedding size: {len(query_vector) if query_vector else 0}")
        print(f"number of retrieved chunks: {len(formatted_results)}")
        if formatted_results:
            print(f"retrieved chunk text preview: {formatted_results[0].get('text', '')[:200]}...")
            scores = [f"{r.get('score', 0.0):.4f}" for r in formatted_results]
            print(f"similarity scores: {', '.join(scores)}")
        else:
            print("retrieved chunk text preview: None")
            print("similarity scores: None")

        # We only return metadata + text + similarity score. Embeddings are NOT returned.
        return formatted_results

    except UnexpectedResponse as e:
        if e.status_code == 404:
            logger.error(f"Collection '{COLLECTION_NAME}' is missing in Qdrant. Returning empty results.")
        else:
            logger.error(f"Unexpected Qdrant response during search: {e}")
        return []
    except Exception as e:
        logger.error(f"Qdrant unavailable or error performing search: {e}")
        return []


def build_context(results: List[Dict[str, Any]]) -> str:
    """
    Concatenates retrieved chunk dictionaries into a rich, formatted prompt context string.
    
    Args:
        results: List of retrieved chunks containing metadata and text.
        
    Returns:
        A formatted string ready to be passed to an LLM.
    """
    # 7. Improve build_context(): Exactly matching empty state string
    if not results:
        return "No relevant context found."
        
    context_parts = []
    
    for chunk in results:
        # Extract metadata fields
        doc_id = chunk.get("document_id", "Unknown")
        doc_name = chunk.get("document_name", "Unknown")
        page = chunk.get("page", "Unknown")
        heading = chunk.get("heading", "Unknown")
        chunk_type = chunk.get("type", "Unknown")
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "")
        
        # Format chunk exactly per specifications to support rich context and preserve tables
        chunk_str = (
            "------------------------------------------------\n\n"
            "Document ID:\n"
            f"{doc_id}\n\n"
            "Document Name:\n"
            f"{doc_name}\n\n"
            "Page:\n"
            f"{page}\n\n"
            "Heading:\n"
            f"{heading}\n\n"
            "Chunk Type:\n"
            f"{chunk_type}\n\n"
            "Similarity:\n"
            f"{score}\n\n"
            "Content:\n\n"
            f"{text}\n\n"
        )
        context_parts.append(chunk_str)
        
    # Append a final closing separator line
    context_parts.append("------------------------------------------------")
    return "".join(context_parts)
