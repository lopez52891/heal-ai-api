import os
import zipfile
import logging
from pathlib import Path
from pypdf import PdfReader

# LangChain + FAISS imports
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ZIP_FILE_NAME = "Gold_Standard.zip"
EXTRACT_TO_DIRECTORY = "gold_standard_docs"
VECTOR_DB_PATH = "faiss_diseases_db"
EMBEDDING_MODEL_NAME = "models/embedding-001"

def unzip_local_file(zip_filename, extract_dir):
    """Unzips a local file if the destination directory doesn't exist."""
    if not os.path.exists(zip_filename):
        logging.error(f"'{zip_filename}' not found in the project directory.")
        return False

    if os.path.exists(extract_dir):
        logging.info(f"Directory '{extract_dir}' already exists. Skipping extraction.")
        return True

    logging.info(f"Unzipping '{zip_filename}' to '{extract_dir}'...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    logging.info("Unzip complete.")
    return True

def main():
    """Main function to build the vector database from a local zip file."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please ensure it's in your .env file.")

    if not unzip_local_file(ZIP_FILE_NAME, EXTRACT_TO_DIRECTORY):
        return

    # 1. Load documents from the extracted directory
    logging.info(f"Loading documents from '{EXTRACT_TO_DIRECTORY}'...")
    docs = []
    # Assumes the zip file contains a directory with text files
    source_dir = Path(EXTRACT_TO_DIRECTORY)
    for file_path in source_dir.rglob("*.pdf"): # Adjust "*.txt" if you have other file types
        try:

            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            if text:
                docs.append(Document(page_content=text, metadata={"source": str(file_path)}))
        except Exception as e:
            logging.warning(f"Could not read file {file_path}: {e}")

    if not docs:
        logging.error("No documents were loaded. Aborting database build.")
        return

    # 2. Initialize embedding model
    logging.info("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=google_api_key)

    # 3. Split documents into chunks
    logging.info(f"Splitting {len(docs)} documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 4. Create and save FAISS vector database
    logging.info(f"Creating FAISS database from {len(chunks)} chunks...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_PATH)

    logging.info(f"Successfully built and saved vector database to '{VECTOR_DB_PATH}'.")


if __name__ == "__main__":
    main()