import logging
import sys
from retrieval_service import search_chunks, build_context

# Configure test logging to WARNING to keep the interactive console output clean and readable
logging.basicConfig(level=logging.WARNING)

def run_interactive_test():
    print("==========================================")
    print("Retrieval Test")
    print("==========================================")

    while True:
        # 1. Interactive Question Input
        query = input("\nEnter your question: ").strip()
        if not query:
            print("Question cannot be empty.")
            continue

        # 2. Optional document_id
        doc_input = input("Enter document_id (leave blank to search entire collection): ").strip()
        document_id = doc_input if doc_input else None

        # 3. Optional top_k
        top_k_input = input("Number of chunks to retrieve (default 3): ").strip()
        try:
            top_k = int(top_k_input) if top_k_input else 3
        except ValueError:
            top_k = 3

        print(f"\nSearching...")
        
        # Perform the search without modifying the architecture
        results = search_chunks(query=query, top_k=top_k, document_id=document_id)

        # 6. Empty Retrieval Handling
        if not results:
            print("\nNo relevant chunks found.")
        else:
            # 4. Better Output Formatting & 9. Production-style Console
            print("\n==========================================")
            print(f"Question: {query}")
            doc_display = document_id if document_id else "All"
            print(f"Document ID: {doc_display}")
            print(f"Retrieved {len(results)} chunks")
            print("==========================================\n")

            for idx, hit in enumerate(results, 1):
                print(f"Chunk {idx}")
                print(f"Score:    {hit.get('score', 0.0):.4f}")
                print(f"Page:     {hit.get('page')}")
                print(f"Heading:  {hit.get('heading')}")
                print(f"Type:     {hit.get('type')}")
                print(f"Chunk ID: {hit.get('chunk_id')}")
                print(f"Tokens:   {hit.get('token_count')}")
                print("\nText:")
                
                text = hit.get('text', '')
                if len(text) > 300:
                    print(text[:300] + "...")
                else:
                    print(text)
                
                print("\n------------------------------------------\n")
                
            # 5. Context Preview
            print("=====================")
            print("Context sent to LLM")
            print("=====================\n")
            
            context_str = build_context(results)
            print(context_str)

        # 7. Loop Mode
        while True:
            again = input("\nSearch again? (y/n): ").strip().lower()
            if again in ['y', 'n']:
                break
            
        if again == 'n':
            print("Exiting retrieval test utility.")
            sys.exit(0)

if __name__ == "__main__":
    try:
        run_interactive_test()
    except KeyboardInterrupt:
        print("\n\nExiting retrieval test utility.")
        sys.exit(0)
