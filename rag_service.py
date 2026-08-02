import logging
import time
import os
from typing import Dict, Any, Optional

from retrieval_service import search_chunks, build_context
from gemini_service import generate_answer
from classifier_service import classify_query

# Use a module-level logger; do not configure basicConfig here
logger = logging.getLogger(__name__)

GEMINI_ERROR_MESSAGE = "An error occurred while generating the answer."
INVALID_QUESTION_MESSAGE = "Please provide a valid question."
UNEXPECTED_ERROR_MESSAGE = "An unexpected error occurred."


def answer_question(question: str, top_k: int = 3, document_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Orchestrates the Retrieval-Augmented Generation (RAG) pipeline to answer a user's question.
    
    This function bridges the retrieval and generation phases without implementing 
    the low-level logic itself. It searches for relevant chunks, formats them into a context,
    and passes them to the language model.
    
    Args:
        question (str): The user's question to answer.
        top_k (int, optional): The number of top relevant chunks to retrieve. Defaults to 3.
        document_id (str, optional): An optional document ID to restrict the search. Defaults to None.
        
    Returns:
        dict: A dictionary containing:
            - success (bool): Whether the process completed without unexpected errors.
            - answer (str): The final generated string or a fallback message.
            - chunks_found (int): The number of chunks successfully retrieved.
            - context_used (bool): True if relevant context was found and passed.
            - document_id (str | None): The document ID used for the search, if any.
            - highest_score (float | None): The highest similarity score among retrieved chunks.
            - retrieval_time (float): Time spent in retrieval.
            - generation_time (float): Time spent in Gemini generation.
            - total_time (float): Total execution time of the RAG pipeline.
    """
    total_start_time = time.time()
    
    # 1. Validate the question
    question = question.strip() if question else ""
    if not question:
        return {
            "success": False,
            "answer": INVALID_QUESTION_MESSAGE,
            "chunks_found": 0,
            "context_used": False
        }

    # 2. Log request received and parameters
    logger.info("Question received.")
    logger.info(f"document_id: {document_id}")
    logger.info(f"top_k: {top_k}")
    
    try:
        logger.info("Classifying query intent.")
        intent = classify_query(question)
        logger.info(f"Query classified as: {intent}")
        
        if intent == "GENERAL":
            logger.info("Skipping retrieval for GENERAL query.")
            generation_start_time = time.time()
            generated_answer = generate_answer(question=question, context="")
            generation_time = time.time() - generation_start_time
            total_time = time.time() - total_start_time
            
            if generated_answer == GEMINI_ERROR_MESSAGE:
                return {
                    "success": False,
                    "answer": GEMINI_ERROR_MESSAGE,
                    "chunks_found": 0,
                    "context_used": False,
                    "document_id": document_id,
                    "highest_score": None,
                    "retrieval_time": 0.0,
                    "generation_time": generation_time,
                    "total_time": total_time
                }
                
            return {
                "success": True,
                "answer": generated_answer,
                "chunks_found": 0,
                "context_used": False,
                "document_id": document_id,
                "highest_score": None,
                "retrieval_time": 0.0,
                "generation_time": generation_time,
                "total_time": total_time
            }

        # Intent is DOCUMENT, proceed with retrieval
        logger.info("Starting retrieval.")
        retrieval_start_time = time.time()
        
        results = search_chunks(
            query=question,
            top_k=top_k,
            document_id=document_id
        )
        
        retrieval_time = time.time() - retrieval_start_time
        num_chunks = len(results)
        
        logger.info("Retrieval completion.")
        logger.info(f"Number of retrieved chunks: {num_chunks}")
        
        highest_score = max((result.get("score", 0.0) for result in results), default=None) if results else None
        
        if highest_score is not None:
            logger.info(f"Highest similarity score: {highest_score}")
        else:
            logger.info("Highest similarity score: N/A")
            
        # Always build context from retrieved chunks and let LLM evaluate relevance
        context = build_context(results)
        
        # 1. Gather global metadata
        upload_folder = "uploads"
        uploaded_docs = []
        if os.path.exists(upload_folder):
            try:
                uploaded_docs = [f for f in os.listdir(upload_folder) if os.path.isfile(os.path.join(upload_folder, f))]
            except Exception as e:
                logger.error(f"Failed to list uploaded documents: {e}")
                
        num_docs = len(uploaded_docs)
        doc_list_str = "\n".join([f"- {d}" for d in uploaded_docs]) if uploaded_docs else "None"
        
        metadata_parts = [
            f"Number of uploaded documents: {num_docs}",
            f"List of uploaded documents:\n{doc_list_str}"
        ]
        
        # 2. Add active document metadata if available
        if document_id:
            metadata_parts.append(f"Active Document ID: {document_id}")
            
        metadata_str = "Document Metadata:\n" + "\n".join(metadata_parts)
        
        # 3. Inject metadata into context block
        if context == "No relevant context found.":
            context = f"{metadata_str}\n\n(Note: No highly relevant text chunks were found for this query.)"
        else:
            context = f"{metadata_str}\n\nRetrieved Chunks:\n{context}"
        
        context_used = bool(results)
        context_len = len(context) if context else 0
        logger.info(f"Context length: {context_len} characters")
        print(f"final Gemini prompt/context length: {context_len}")
        print("==============================\n")

        # 6. Pass question and context to Gemini
        # We always call generate_answer; gemini_service handles empty contexts cleanly
        logger.info("Gemini generation start...")
        generation_start_time = time.time()
        
        generated_answer = generate_answer(question=question, context=context)
        
        generation_time = time.time() - generation_start_time
        total_time = time.time() - total_start_time
        logger.info(f"Total execution time: {total_time:.3f} seconds.")
        
        # Detect generation failure
        if generated_answer == GEMINI_ERROR_MESSAGE:
            logger.error("generation failed.")
            logger.info("Status: failure")
            return {
                "success": False,
                "answer": GEMINI_ERROR_MESSAGE,
                "chunks_found": num_chunks,
                "context_used": context_used,
                "document_id": document_id,
                "highest_score": highest_score,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time
            }
            
        logger.info("Status: success")
        
        # Return the final structured payload for a valid answer
        return {
            "success": True,
            "answer": generated_answer,
            "chunks_found": num_chunks,
            "context_used": context_used,
            "document_id": document_id,
            "highest_score": highest_score,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time
        }

    except Exception as e:
        # Catch unexpected exceptions gracefully
        logger.error(f"An unexpected error occurred during RAG orchestration: {e}")
        logger.info("Status: failure")
        
        return {
            "success": False,
            "answer": UNEXPECTED_ERROR_MESSAGE,
            "chunks_found": 0,
            "context_used": False,
            "highest_score": None,
            "retrieval_time": 0.0,
            "generation_time": 0.0,
            "total_time": time.time() - total_start_time
        }
