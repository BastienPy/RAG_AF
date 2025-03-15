import os

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if HUGGINGFACE_API_KEY:
    print("API key retrieved successfully!")
else:
    print("API key not found. Make sure it's set in your environment.")

HUGGINGFACE_MODEL_NAME = "bigscience/bloom-560m"

CHROMA_PERSIST_DIR = "./chroma_db"

CHROMA_COLLECTION_NAME = "financial_analysis"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

FINANCIAL_REPORTS_DIR = "./financial_reports"