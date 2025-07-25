import os
import pypdf
import io
from pathlib import Path

# --- Configuration ---
PATIENT_FILES_DIR = Path("patient_files")
PATIENT_FILES_DIR.mkdir(exist_ok=True)

# In-memory "database" for conversation histories
conversation_histories = {}


class PatientService:
    def save_pdf_as_text(self, patient_id: str, file_content: bytes) -> bool:
        """Extracts text from an uploaded PDF and saves it as a .txt file."""
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            file_path = PATIENT_FILES_DIR / f"{patient_id}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
            return True
        except Exception:
            return False

    def get_patient_history_text(self, patient_id: str) -> str | None:
        """Reads the full text history from a patient's .txt file."""
        file_path = PATIENT_FILES_DIR / f"{patient_id}.txt"
        if not file_path.exists():
            return None

        return file_path.read_text(encoding="utf-8")

    def append_to_patient_history(self, patient_id: str, new_text: str) -> bool:
        """Appends a new note to the patient's history .txt file."""
        file_path = PATIENT_FILES_DIR / f"{patient_id}.txt"
        if not file_path.exists():
            return False

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- Appended Note ---\n{new_text}")
        return True

    def get_conversation_history(self, patient_id: str) -> list[str]:
        """Retrieves the conversation history for a patient."""
        return conversation_histories.get(patient_id, [])

    def add_to_conversation_history(self, patient_id: str, user_query: str, agent_answer: str):
        """Adds a new turn to the conversation history."""
        if patient_id not in conversation_histories:
            conversation_histories[patient_id] = []

        conversation_histories[patient_id].append(f"User: {user_query}")
        conversation_histories[patient_id].append(f"Agent: {agent_answer}")