import logging
from qdrant_service import create_collection, store_chunks, delete_document, count_vectors

# Configure test logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_tests():
    print("========================================")
    print("Testing Qdrant Storage Module")
    print("========================================")

    # 1. Create two sample embedded chunks
    test_document_id = "test_doc_001"
    
    # Using 384-dimensional mock embeddings
    mock_embedding_1 = [0.01] * 384
    mock_embedding_2 = [0.02] * 384
    
    chunks = [
        {
            "document_id": test_document_id,
            "page": 1,
            "chunk": 1,
            "chunk_id": f"{test_document_id}_page1_chunk1",
            "type": "paragraph",
            "heading": "Introduction",
            "heading_level": 1,
            "token_count": 240,
            "text": "This is the first sample text chunk.",
            "embedding": mock_embedding_1
        },
        {
            "document_id": test_document_id,
            "page": 1,
            "chunk": 2,
            "chunk_id": f"{test_document_id}_page1_chunk2",
            "type": "paragraph",
            "heading": "Introduction",
            "heading_level": 1,
            "token_count": 180,
            "text": "This is the second sample text chunk.",
            "embedding": mock_embedding_2
        }
    ]
    
    try:
        # 2. Create the collection if needed
        print("\n--- Creating Collection ---")
        create_collection()
        print("Collection created")
        
        # 3. Store both chunks
        print("\n--- Storing Vectors ---")
        # We need to make a copy because store_chunks removes the 'embedding' key from the dict
        chunks_to_insert = [chunk.copy() for chunk in chunks]
        store_chunks(chunks_to_insert)
        print("Vectors inserted")
        
        # 4. Print current vector count
        print("\n--- Vector Count ---")
        current_count = count_vectors()
        print(f"Current vector count: {current_count}")
        
        # 5. Delete the sample document
        print("\n--- Deleting Document ---")
        delete_document(test_document_id)
        
        # 6. Print the vector count again
        print("\n--- Final Vector Count ---")
        final_count = count_vectors()
        print(f"Current vector count: {final_count}")
        
        print("\n========================================")
        print("Tests completed successfully")
        print("========================================")

    except Exception as e:
        print(f"\nTest failed with error: {e}")

if __name__ == "__main__":
    run_tests()
