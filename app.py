# ==========================================================
# FastAPI imports
# ==========================================================

# FastAPI framework
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# File/folder operations ke liye
import os

from pdf_utils import pdf_to_images
from rag_service import answer_question
from ingestion_service import ingest_pdf


# ==========================================================
# FastAPI Application
# ==========================================================

# FastAPI application create kar rahe hain
app = FastAPI(title="Nutrition OCR Chatbot")


# ==========================================================
# Folder Configuration
# ==========================================================

# Uploaded PDFs is folder mein save hongi
UPLOAD_FOLDER = "uploads"

# PDF se bani images is folder mein save hongi
OUTPUT_FOLDER = "output"

# Agar folders exist nahi karte to automatically create kar do
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files for CSS and JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates directory
templates = Jinja2Templates(directory="templates")

class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    document_id: str = None


# ==========================================================
# Home API / Web Interface
# ==========================================================

# Serve the chatbot Web UI
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ==========================================================
# Chat API
# ==========================================================

@app.post("/ask")
def ask_question(request_data: AskRequest):
    # Call the existing RAG orchestrator directly
    return answer_question(
        question=request_data.question,
        top_k=request_data.top_k,
        document_id=request_data.document_id
    )


# ==========================================================
# Upload PDF API
# ==========================================================

# User PDF upload karega
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # ------------------------------------------------------
    # Step 1 : Uploaded PDF ka path banao
    # Example:
    # uploads/pizza.pdf
    # ------------------------------------------------------

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    # ------------------------------------------------------
    # Step 2 : PDF ko uploads folder mein save karo
    # "wb" = write binary
    # ------------------------------------------------------

    with open(file_path, "wb") as f:
        f.write(await file.read())

    ingestion_result = ingest_pdf(file_path, file.filename)

    if not ingestion_result:
        return {"success": False, "message": "Failed to ingest document"}

    return {
        "success": True,
        "message": "File uploaded and processed successfully",
        "document_name": ingestion_result.get("document_name"),
        "document_id": ingestion_result.get("document_id"),
        "chunk_count": ingestion_result.get("total_chunks"),
        "embedding_count": ingestion_result.get("total_vectors")
    }