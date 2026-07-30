import sys
import logging
from rag_service import answer_question

# Suppress logs from underlying services to keep the chat UI completely clean
logging.basicConfig(level=logging.WARNING)

def run_chat():
    """
    Runs an interactive command-line loop allowing the user to test the 
    end-to-end RAG pipeline (Retrieval-Augmented Generation) with performance metrics.
    """
    print("========================================")
    print("Nutrition OCR RAG Chat")
    print("========================================")
    
    while True:
        try:
            print("\nQuestion:")
            question = input("").strip()
            
            # Check for exit conditions
            if question.lower() in ["exit", "quit"]:
                print("Exiting chat. Goodbye!")
                break
                
            if not question:
                continue
                
            doc_input = input("Document ID (press Enter for all documents): ").strip()
            document_id = doc_input if doc_input else None
            
            top_k_input = input("Top K (default 3): ").strip()
            try:
                top_k = int(top_k_input) if top_k_input else 3
            except ValueError:
                top_k = 3
                
            print("\nSearching...\n")
            
            # Execute the RAG pipeline orchestration
            response = answer_question(
                question=question,
                top_k=top_k,
                document_id=document_id
            )
            
            # Format and display the results strictly as requested
            print("Answer:")
            if response.get("success"):
                print(response.get("answer"))
            else:
                print(f"Error: {response.get('answer')}")
                
            print("\n----------------------------------------\n")
            print("Retrieval Metrics")
            print("-----------------")
            
            chunks_retrieved = response.get("chunks_found", 0)
            highest_score = response.get("highest_score")
            
            print(f"Chunks Retrieved : {chunks_retrieved}")
            if chunks_retrieved == 0 or highest_score is None:
                print("Highest Score    : N/A")
            else:
                print(f"Highest Score    : {highest_score:.4f}")
                
            print("\nPerformance")
            print("-----------")
            print(f"Retrieval Time  : {response.get('retrieval_time', 0.0):.3f} s")
            print(f"Generation Time : {response.get('generation_time', 0.0):.3f} s")
            print(f"Total Time      : {response.get('total_time', 0.0):.3f} s")
            
            print("\nDocument:")
            print(document_id if document_id else "All Documents")
            
            print("\n========================================")
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting chat. Goodbye!")
            sys.exit(0)
        except Exception as e:
            # Only use logging for unexpected script-level failures
            logging.error(f"An unexpected failure occurred in the CLI loop: {e}")
            print("\nAn unexpected error occurred. Please try again.")

if __name__ == "__main__":
    run_chat()
