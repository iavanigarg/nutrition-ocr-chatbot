import logging
import uuid
import os
import shutil
from typing import Dict, Any, Optional

# Import existing services without modification
from pdf_utils import pdf_to_images
from ocr_service import extract_text
from chunking_service import chunk_page
from embedding_service import embed_chunks
from qdrant_service import store_chunks, create_collection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest_pdf(pdf_path: str, document_name: str = "Unknown") -> Optional[Dict[str, Any]]:
    """
    Ingests a single PDF document by processing it page-by-page.
    Extracts text via OCR, chunks it, embeds it, and stores it in Qdrant.
    
    Args:
        pdf_path: The file path to the PDF to be ingested.
        
    Returns:
        A dictionary containing the ingestion summary, or None if the process fails to start.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at: {pdf_path}")
        return None

    # Generate one unique document_id for the entire PDF
    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    logger.info(f"Starting ingestion for '{pdf_path}' with document_id: '{document_id}'")

    # Ensure collection exists before storing
    try:
        create_collection()
    except Exception as e:
        logger.error(f"Failed to verify or create Qdrant collection: {e}")
        return None

    # Temporary output folder for storing extracted page images
    output_folder = os.path.join("output", document_id)
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Convert the PDF into images page-by-page using the existing utility
        logger.info("Converting PDF to images...")
        image_paths = pdf_to_images(pdf_path, output_folder)
        
        total_pages = len(image_paths)
        logger.info(f"Successfully converted PDF into {total_pages} images.")

        total_chunks = 0
        total_vectors = 0
        total_extracted_text_length = 0
        first_chunk_preview = "None"
        embedding_dimension = 0

        # Process one page at a time to keep memory usage low
        for idx, image_path in enumerate(image_paths):
            # Maintain page numbers correctly
            page_number = idx + 1
            
            # Display progress
            print(f"\nProcessing page {page_number}/{total_pages}...")

            try:
                # OCR extraction
                page_text = extract_text(image_path)
                
                # Check for empty OCR output
                if not page_text or not page_text.strip():
                    logger.warning(f"OCR returned empty text for page {page_number}. Skipping page.")
                    print(f"OCR output empty. Skipping page {page_number}.")
                    continue
                    
                total_extracted_text_length += len(page_text)
                print("OCR completed")
                
                # Chunking
                chunks = chunk_page(
                    page_text=page_text, 
                    page_number=page_number, 
                    document_id=document_id,
                    document_name=document_name
                )
                
                # Check for empty chunk lists
                if not chunks:
                    logger.warning(f"Chunking returned no chunks for page {page_number}. Skipping.")
                    continue
                    
                print(f"Created {len(chunks)} chunks")
                
                # Generate embeddings
                embedded_chunks = embed_chunks(chunks)
                print("Embeddings generated")
                
                if first_chunk_preview == "None" and chunks:
                    first_chunk_preview = chunks[0].get("text", "")[:100].replace("\n", " ") + "..."
                
                # Validate embeddings before storage
                valid_chunks = []
                for chunk in embedded_chunks:
                    emb = chunk.get("embedding")
                    if embedding_dimension == 0 and emb:
                        embedding_dimension = len(emb)
                    # Check that embedding exists, is not empty, and length is exactly 384
                    if emb and isinstance(emb, list) and len(emb) == 384:
                        valid_chunks.append(chunk)
                    else:
                        logger.warning(f"Chunk {chunk.get('chunk_id')} has invalid or missing embedding. Skipping storage for this chunk.")
                        
                if valid_chunks:
                    # Store only valid vectors
                    store_chunks(valid_chunks)
                    print(f"Stored {len(valid_chunks)} vectors")
                else:
                    logger.warning(f"No valid embedded chunks found for page {page_number}. Skipping storage.")
                    print(f"Stored 0 vectors")
                
                # Maintain running totals (Only counting successfully stored vectors)
                total_chunks += len(chunks)
                total_vectors += len(valid_chunks)

            except Exception as e:
                # If one page fails OCR, log the error and continue
                logger.error(f"Failed to process page {page_number}: {e}")
                print(f"Error processing page {page_number}. Skipping to next page.")
                continue

        # Print final summary exactly as before
        print("\n========================================")
        print("INGESTION SUMMARY")
        print("========================================")
        print(f"Document ID:   {document_id}")
        print(f"Total Pages:   {total_pages}")
        print(f"Total Chunks:  {total_chunks}")
        print(f"Total Vectors: {total_vectors}")
        print("========================================\n")
        
        from qdrant_service import count_vectors
        current_point_count = count_vectors()
        
        print("\n=== DEBUG LOGS: FILE UPLOAD ===")
        print(f"filename: {os.path.basename(pdf_path)}")
        print(f"file size: {os.path.getsize(pdf_path)} bytes")
        print(f"extracted text length: {total_extracted_text_length}")
        print(f"number of chunks: {total_chunks}")
        print(f"first chunk preview: {first_chunk_preview}")
        print(f"embedding dimension: {embedding_dimension}")
        print(f"Qdrant point count: {current_point_count}")
        print("===============================\n")
        
        # Return ingestion summary dictionary
        return {
            "document_id": document_id,
            "document_name": document_name,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "total_vectors": total_vectors
        }

    finally:
        # Clean up temporary images to save space
        if os.path.exists(output_folder):
            try:
                shutil.rmtree(output_folder)
                logger.info(f"Cleaned up temporary image folder: {output_folder}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary folder {output_folder}: {e}")
