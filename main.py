from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import traceback
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import json
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import boto3 
from botocore.exceptions import ClientError
import httpx    
from pydantic import BaseModel 
import asyncio # Make sure asyncio is imported

# --- Logging ---
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Initialize Clients ---
app = FastAPI(title="Document Processor")

try:
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
    )
    s3_client = boto3.client(
        's3',
        region_name=os.getenv("AWS_S3_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
    NEXTJS_CALLBACK_URL = os.getenv("NEXTJS_CALLBACK_URL")
    PYTHON_SERVICE_SECRET = os.getenv("PYTHON_SERVICE_SECRET")

    # Validate essential environment variables
    if not all([os.getenv("OPENAI_API_KEY"), os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"), S3_BUCKET_NAME, NEXTJS_CALLBACK_URL, PYTHON_SERVICE_SECRET]):
        logger.critical("CRITICAL: Missing one or more required environment variables!")
        # You might want to exit here in a real deployment
        # sys.exit("Missing environment variables")

except Exception as e:
    logger.critical(f"CRITICAL: Failed to initialize clients or load environment variables: {e}")
    # sys.exit("Initialization failed")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TODO: Restrict this in production to your Next.js app's URL
    allow_credentials=True,
    allow_methods=["POST", "GET"], # Limit methods if needed
    allow_headers=["*"],
)

# --- Pydantic Model for Request Body ---
class ProcessRequest(BaseModel):
    s3_key: str
    document_id: str
    # Add other fields if Next.js sends them (e.g., original filename for metadata)
    # filename: Optional[str] = None

# --- Helper Functions ---
def extract_text_with_pypdf(file_path):
    """Extract text using PyPDF as a fallback method"""
    logger.info(f"Attempting text extraction with PyPDF for {file_path}")
    text = ""
    try:
        # Use 'with' statement to ensure the file is closed properly
        with open(file_path, 'rb') as f: # Open in binary mode for pypdf
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n" # Add separator between pages
        logger.info(f"PyPDF extracted approx {len(text)} characters.")
        return text
    except Exception as e:
        logger.error(f"PyPDF extraction error: {str(e)}")
        logger.error(traceback.format_exc()) # Log full traceback for debugging
        return ""

async def notify_nextjs(document_id: str, success: bool, error_message: Optional[str] = None):
    """Sends completion status back to the Next.js API route with retries."""
    if not NEXTJS_CALLBACK_URL or not PYTHON_SERVICE_SECRET:
        logger.error("Callback URL or Secret not configured. Cannot notify Next.js.")
        return

    payload = {
        "document_id": document_id,
        "success": success,
        "error_message": error_message if not success else None
    }
    headers = {
        "Content-Type": "application/json",
        "X-Processing-Secret": PYTHON_SERVICE_SECRET # Authentication header
    }

    # --- Retry Logic Parameters ---
    max_retries = 3
    base_delay = 2  # seconds
    connect_timeout = 15.0 # Increased connect timeout
    read_timeout = 30.0  # Standard read timeout
        # Provide all four timeout values: connect, read, write, pool
    timeout_config = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=read_timeout, pool=read_timeout)
    # --- End Retry Logic Parameters ---


    for attempt in range(max_retries):
        logger.info(f"Attempting to notify Next.js (Attempt {attempt + 1}/{max_retries}) at {NEXTJS_CALLBACK_URL} for doc {document_id}, success={success}")
        # Use a new client for each attempt to ensure clean state for timeouts
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            try:
                response = await client.post(NEXTJS_CALLBACK_URL, json=payload, headers=headers)
                response.raise_for_status() # Check for 4xx/5xx errors
                logger.info(f"Successfully notified Next.js for document {document_id} after attempt {attempt + 1}. Status: {success}. Response: {response.status_code}")
                return # Exit loop on success

            except httpx.ConnectTimeout as e:
                logger.warning(f"ConnectTimeout on attempt {attempt + 1} for doc {document_id}: {e}. Retrying in {base_delay * (2**attempt)}s...")
                if attempt == max_retries - 1: # Last attempt failed
                    logger.error(f"ConnectTimeout failed after {max_retries} attempts for doc {document_id}.")
                    logger.error(traceback.format_exc())
                    return # Exit after final attempt
                await asyncio.sleep(base_delay * (2**attempt)) # Exponential backoff

            except httpx.ReadTimeout as e: # Handle potential read timeouts too
                logger.warning(f"ReadTimeout on attempt {attempt + 1} for doc {document_id}: {e}. Retrying in {base_delay * (2**attempt)}s...")
                if attempt == max_retries - 1: # Last attempt failed
                    logger.error(f"ReadTimeout failed after {max_retries} attempts for doc {document_id}.")
                    logger.error(traceback.format_exc())
                    return # Exit after final attempt
                await asyncio.sleep(base_delay * (2**attempt)) # Exponential backoff

            except httpx.RequestError as e: # Other non-timeout request errors (likely not retryable)
                logger.error(f"HTTP Request error (non-timeout) notifying Next.js for document {document_id}: {e}")
                logger.error(traceback.format_exc())
                return # Exit loop on non-retryable request errors

            except httpx.HTTPStatusError as e: # Handle 4xx/5xx errors
                # Retry on 5xx errors (server-side issues)
                if 500 <= e.response.status_code < 600 and attempt < max_retries - 1:
                     logger.warning(f"HTTP Status error {e.response.status_code} on attempt {attempt + 1} for doc {document_id}. Retrying in {base_delay * (2**attempt)}s...")
                     await asyncio.sleep(base_delay * (2**attempt)) # Exponential backoff
                     continue # Go to next attempt in the loop

                # Log final 5xx failure or any 4xx error and exit
                logger.error(f"HTTP Status error notifying Next.js for document {document_id} (final attempt or non-retryable): {e.response.status_code} - {e.response.text}")
                return # Exit loop on non-retryable status errors (e.g., 4xx) or final 5xx failure

            except Exception as e: # Catch-all for unexpected errors during the request
                logger.error(f"Unexpected error notifying Next.js for document {document_id} during attempt {attempt + 1}: {e}")
                logger.error(traceback.format_exc())
                return # Exit loop on unexpected errors

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/process")
async def process_document(payload: ProcessRequest):
    """
    Download doc from S3, process into chunks/embeddings,
    upload results to S3, and trigger indexing via Next.js callback.
    """
    s3_key = payload.s3_key
    document_id = payload.document_id
    # Extract original filename from key if needed, assuming format folder/uuid-filename.ext
    original_filename = s3_key.split('-')[-1] if '-' in s3_key else s3_key.split('/')[-1]

    logger.info(f"Received processing request for document_id: {document_id}, s3_key: {s3_key}")

    temp_path = None
    try:
        # 1. Download file from S3
        _, ext = os.path.splitext(s3_key)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".tmp") as temp:
            temp_path = temp.name
            logger.info(f"Downloading s3://{S3_BUCKET_NAME}/{s3_key} to {temp_path}")
            try:
                s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_path)
                logger.info(f"S3 download successful for {document_id}.")
            except ClientError as e:
                 error_code = e.response.get("Error", {}).get("Code")
                 error_message = f"S3 Download error: {e}"
                 if error_code == '404': # Check for Not Found specifically
                     error_message = f"S3 Key not found: {s3_key}"
                     logger.error(error_message)
                     await notify_nextjs(document_id, success=False, error_message=error_message)
                     raise HTTPException(status_code=404, detail=error_message)
                 else:
                     logger.error(error_message)
                     await notify_nextjs(document_id, success=False, error_message=error_message)
                     raise HTTPException(status_code=500, detail=error_message)

        # 2. Extract Text & Chunk
        logger.info(f"Processing downloaded file: {temp_path} for doc: {document_id}")
        documents: List[Document] = []
        # Base metadata to add to all chunks
        base_metadata = {"document_id": document_id, "original_filename": original_filename}
        # Add folderId if it's part of the s3_key structure (e.g., "folderId/uuid-name.pdf")
        key_parts = s3_key.split('/')
        if len(key_parts) > 1 and key_parts[0] != 'root':
             base_metadata["folder_id"] = key_parts[0]
        else:
             base_metadata["folder_id"] = "root" # Explicitly mark root


        # --- MINIMAL CHANGE START ---
        # Introduce a flag to easily switch between strategies later if needed
        # Set to False to always skip Unstructured and use PyPDF fallback
        USE_UNSTRUCTURED_LOADER = False

        if USE_UNSTRUCTURED_LOADER:
            # Try Unstructured (consider making strategy configurable via payload if needed)
            try:
                # Change strategy here if you want to re-enable Unstructured later
                loader = UnstructuredLoader(temp_path, strategy="hi_res") # or "fast"
                loaded_docs = loader.load() # Unstructured chunks internally
                if loaded_docs:
                     # Add base metadata to unstructured chunks
                     for doc in loaded_docs:
                         doc.metadata.update(base_metadata)
                     documents = loaded_docs
                     logger.info(f"Unstructured loaded {len(documents)} chunks for doc: {document_id}.")
                else:
                     logger.info(f"UnstructuredLoader returned no chunks for doc: {document_id}.")
            except Exception as loader_error:
                logger.warning(f"UnstructuredLoader failed for doc {document_id}: {loader_error}. Trying PyPDF.")
                # If Unstructured fails, 'documents' remains empty, forcing the fallback below.
        # --- MINIMAL CHANGE END ---


        # Fallback to PyPDF (This will now always run if USE_UNSTRUCTURED_LOADER is False)
        # If USE_UNSTRUCTURED_LOADER was False, 'documents' is still [], so this block executes.
        # If USE_UNSTRUCTURED_LOADER was True but failed or returned no chunks, this block also executes.
        if not documents:
            extracted_text_pypdf = extract_text_with_pypdf(temp_path)
            if extracted_text_pypdf:
                logger.info(f"PyPDF extracted text for doc {document_id}. Splitting...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150,
                    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
                    length_function=len,
                )
                # Create a single Document object with base metadata for splitting
                full_doc_obj = Document(page_content=extracted_text_pypdf, metadata=base_metadata)
                documents = text_splitter.split_documents([full_doc_obj])
                logger.info(f"Split into {len(documents)} chunks using text splitter for doc: {document_id}.")
            else:
                logger.warning(f"PyPDF fallback also failed for doc {document_id}. No text extracted.")
                await notify_nextjs(document_id, success=False, error_message="Failed to extract text from document.")
                # Decide: Raise error or allow "empty" success? Let's raise for clarity.
                raise HTTPException(status_code=400, detail="Failed to extract text from document.")

        # 3. Prepare for Embedding & S3 Upload of Results
        chunks_for_json = []
        texts_to_embed = []

        if not documents:
             logger.warning(f"No document chunks generated for doc: {document_id}.")
             await notify_nextjs(document_id, success=True, error_message="Document processed but contained no text chunks.")
             return {"message": f"Document {document_id} processed, no text chunks found."}

        for i, doc in enumerate(documents):
            chunk_text = doc.page_content
            # Metadata already contains document_id, folder_id, filename from base_metadata
            chunk_metadata = doc.metadata
            chunk_metadata["chunk_index"] = i # Add chunk index

            # Pinecone ID: Ensure it's unique and valid ASCII
            # Consider sanitizing if document_id or i could be problematic, though cuid + index should be safe
            chunk_id = f"{document_id}_chunk_{i}"

            chunks_for_json.append({
                "id": chunk_id,              # ID for the Pinecone vector
                "text": chunk_text,          # Text content of the chunk
                "metadata": chunk_metadata,  # Metadata for Pinecone vector
            })
            texts_to_embed.append(chunk_text)

        # 4. Generate Embeddings
        logger.info(f"Generating embeddings for {len(texts_to_embed)} chunks for doc: {document_id}...")
        try:
            embeddings_list = embeddings.embed_documents(texts_to_embed)
            logger.info(f"Embedding generation complete for doc: {document_id}.")
        except Exception as embed_error:
            logger.error(f"Embedding generation failed for doc {document_id}: {embed_error}")
            await notify_nextjs(document_id, success=False, error_message=f"Embedding generation failed: {embed_error}")
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {embed_error}")

        # 5. Upload Processed Data (JSON with chunks/embeddings) back to S3
        processed_data = {
            "document_id": document_id,
            "chunks_with_metadata": chunks_for_json, # Contains ID, text, metadata
            "embeddings": embeddings_list            # List of vectors
        }
        processed_data_key = f"processed/{document_id}.json"
        logger.info(f"Uploading processed data for doc {document_id} to s3://{S3_BUCKET_NAME}/{processed_data_key}")
        try:
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=processed_data_key,
                Body=json.dumps(processed_data), # Serialize dict to JSON string
                ContentType='application/json'
            )
            logger.info(f"Successfully uploaded processed data for doc {document_id}")

            # 6. Notify Next.js of SUCCESS (Now Next.js knows where to get data)
            await notify_nextjs(document_id, success=True)

            return {"message": f"Processing complete for {document_id}. Processed data at {processed_data_key}"}

        except ClientError as e:
            error_message = f"Failed to upload processed data to S3 for doc {document_id}: {e}"
            logger.error(error_message)
            await notify_nextjs(document_id, success=False, error_message=error_message)
            raise HTTPException(status_code=500, detail=error_message)

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like 404, 400) to return correct status codes
        raise http_exc
    except Exception as e:
        # Catch-all for unexpected errors during the whole process
        error_message = f"Unexpected error processing document {document_id}: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        await notify_nextjs(document_id, success=False, error_message=error_message)
        raise HTTPException(status_code=500, detail=error_message)
    finally:
        # Clean up downloaded temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Temporary file {temp_path} removed for doc {document_id}")
            except Exception as cleanup_error:
                 logger.error(f"Error removing temp file {temp_path} for doc {document_id}: {cleanup_error}")

# --- Run the App ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000)) # Default to 8000 if PORT not set
    logger.info(f"Starting Uvicorn server on 0.0.0.0:{port}")
    # Use reload=True only for local development
    # uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False) # For deployment