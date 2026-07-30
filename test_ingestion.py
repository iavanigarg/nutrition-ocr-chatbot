import sys
import os
from ingestion_service import ingest_pdf

def run_test():
    print("========================================")
    print("Testing Ingestion Pipeline")
    print("========================================")
    
    # Default to a sample file, but accept CLI argument
    test_pdf_path = "sample.pdf"
    
    # Accept PDF path from arguments
    if len(sys.argv) > 1:
        test_pdf_path = sys.argv[1]
        
    if not os.path.exists(test_pdf_path):
        print(f"\nError: PDF file '{test_pdf_path}' not found.")
        print("\nPlease provide a valid PDF path as an argument:")
        print("Usage: python test_ingestion.py <path_to_your_file.pdf>")
        return
        
    print(f"\nStarting ingestion for: {test_pdf_path}")
    
    # Run the complete ingestion pipeline
    ingest_pdf(test_pdf_path)

if __name__ == "__main__":
    run_test()
