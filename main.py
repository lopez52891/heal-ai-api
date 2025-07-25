from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import uuid
import io

from dotenv import load_dotenv
load_dotenv()

from rag_service import RAGService
from patient_service import PatientService

# Initialize the FastAPI app and services
app = FastAPI(
    title="Heal.ai",
    description="An AI-powered clinical co-pilot featuring PDF uploads and conversation memory.",
    version="3.0.0"
)

try:
    rag_service = RAGService()
    patient_service = PatientService()
except Exception as e:
    raise RuntimeError(f"FATAL: Failed to initialize services. Application cannot start. Error: {e}")


# --- API Data Models ---
class PatientQueryRequest(BaseModel):
    query: str = Field(..., description="The user's latest question or prompt.")

class AppendHistoryRequest(BaseModel):
    text: str = Field(..., description="The new text note to append to the patient's history.")

class SourceDocument(BaseModel):
    source: str | None
    url: str | None
    title: str | None

class RAGQueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]

class PDFUploadResponse(BaseModel):
    patient_id: str
    filename: str
    info: str

class AppendHistoryResponse(BaseModel):
    patient_id: str
    info: str

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "Medical RAG API is running."}

@app.post("/api/v1/patients/upload", response_model=PDFUploadResponse)
async def upload_patient_pdf(file: UploadFile = File(...)):
    """
    Uploads a patient's PDF, extracts the text, and saves it as a .txt file.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    patient_id = str(uuid.uuid4())
    file_content = await file.read()

    success = patient_service.save_pdf_as_text(patient_id, file_content)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process and save the uploaded PDF.")

    return {
        "patient_id": patient_id,
        "filename": file.filename,
        "info": f"File processed and saved as text. Use the patient_id for queries."
    }

@app.post("/api/v1/patients/{patient_id}/append", response_model=AppendHistoryResponse)
async def append_to_history(patient_id: str, request: AppendHistoryRequest):
    """
    Appends a new text note to an existing patient's history.
    """
    success = patient_service.append_to_patient_history(patient_id, request.text)
    if not success:
        raise HTTPException(status_code=404, detail=f"Patient with ID '{patient_id}' not found.")

    return {
        "patient_id": patient_id,
        "info": "The patient's history has been successfully updated."
    }

@app.post("/api/v1/patients/{patient_id}/query", response_model=RAGQueryResponse)
async def query_patient_agent(patient_id: str, request: PatientQueryRequest):
    """
    Queries the agent about a specific patient, using conversation memory.
    """
    patient_history = patient_service.get_patient_history_text(patient_id)
    if patient_history is None:
        raise HTTPException(status_code=404, detail=f"Patient with ID '{patient_id}' not found.")

    conversation_history = patient_service.get_conversation_history(patient_id)

    result = rag_service.process_query(
        patient_history=patient_history,
        conversation_history=conversation_history,
        query=request.query
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    patient_service.add_to_conversation_history(patient_id, request.query, result["answer"])

    return result