"""
Test script for the embedding service.
Validates the initialization of the embedding model and generation of embeddings.
"""

from embedding_service import embed_chunks

def test_embedding_module():
    print("--- Starting Embedding Service Test ---")
    
    # 1. Create two sample chunks
    sample_chunks = [
        {
            "document_id": "doc_001",
            "page": 1,
            "chunk": 1,
            "chunk_id": "doc_001_p001_c001",
            "type": "heading",
            "heading": "Introduction",
            "heading_level": 1,
            "token_count": 5,
            "text": "Introduction"
        },
        {
            "document_id": "doc_001",
            "page": 1,
            "chunk": 2,
            "chunk_id": "doc_001_p001_c002",
            "type": "paragraph",
            "heading": "Introduction",
            "heading_level": 1,
            "token_count": 25,
            "text": "This is a sample paragraph describing the nutrition OCR chatbot system. It aims to process dietary logs accurately."
        }
    ]

    print(f"\n1. Input: Created {len(sample_chunks)} sample chunks.")

    # 2. Call embed_chunks()
    print("\n2. Processing chunks through embed_chunks()...")
    processed_chunks = embed_chunks(sample_chunks)

    # 3. Print outputs
    print("\n3. Output Results:")
    print("-" * 30)
    
    # Number of chunks processed
    print(f"Number of chunks processed: {len(processed_chunks)}")
    
    if len(processed_chunks) > 0 and "embedding" in processed_chunks[0]:
        first_chunk = processed_chunks[0]
        embedding_vec = first_chunk["embedding"]
        
        # Embedding dimension
        print(f"Embedding dimension: {len(embedding_vec)}")
        
        # First 5 embedding values
        print(f"First 5 embedding values: {embedding_vec[:5]}")
        
        # Chunk metadata (print one of them excluding the full embedding for readability)
        print("\nChunk metadata (First chunk):")
        for key, value in first_chunk.items():
            if key == "embedding":
                print(f"  {key}: [List of {len(value)} floats]")
            else:
                print(f"  {key}: {value}")
    else:
        print("Error: Embeddings were not generated.")
        
    print("-" * 30)
    print("--- Test Completed ---")

if __name__ == "__main__":
    test_embedding_module()
