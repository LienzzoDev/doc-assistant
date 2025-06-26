from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import traceback
import logging
import asyncio
import signal
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import json
import boto3 
from botocore.exceptions import ClientError
import httpx    
from pydantic import BaseModel 
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.document import DoclingDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import re
from openai import OpenAI
import base64
from pathlib import Path

# --- Logging ---
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CRITICAL: Set up HuggingFace cache paths BEFORE any imports ---
# This ensures docling uses the pre-cached models
os.environ['HF_HOME'] = '/app/.cache/huggingface'
os.environ['HF_HUB_CACHE'] = '/app/.cache/huggingface/hub'
os.environ['TRANSFORMERS_CACHE'] = '/app/.cache/huggingface/transformers'
os.environ['HF_DATASETS_CACHE'] = '/app/.cache/huggingface/datasets'

# Force offline mode to use cached models only
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Log cache status at startup
def verify_model_cache():
    """Verify that models are cached and accessible"""
    cache_path = Path('/app/.cache/huggingface/hub')
    
    if cache_path.exists():
        file_count = len([f for f in cache_path.rglob('*') if f.is_file()])
        logger.info(f"Model cache found at {cache_path} with {file_count} files")
        
        # Look for docling models specifically
        docling_models = list(cache_path.glob('**/models--ds4sd--docling-models*'))
        if docling_models:
            logger.info(f"Found docling models: {[str(p) for p in docling_models]}")
            return True
        else:
            logger.warning("Docling models not found in cache!")
            return False
    else:
        logger.error(f"Model cache directory does not exist: {cache_path}")
        return False

# Verify cache on startup
cache_available = verify_model_cache()
if not cache_available:
    logger.critical("CRITICAL: Model cache not available - this will cause runtime failures!")

# --- Load Environment Variables ---
load_dotenv()

# --- Timeout Handler ---
def timeout_handler(signum, frame):
    raise TimeoutError("Docling processing timed out")

# --- Initialize Clients ---
app = FastAPI(title="Document Processor")

try:
    s3_client = boto3.client(
        's3',
        region_name=os.getenv("AWS_S3_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
    NEXTJS_CALLBACK_URL = os.getenv("NEXTJS_CALLBACK_URL")
    PYTHON_SERVICE_SECRET = os.getenv("PYTHON_SERVICE_SECRET")
    PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "__default__")

    # Initialize Docling converter with OPTIMIZED table parsing options
    logger.info("Initializing Docling converter...")
    
    pipeline_options = PdfPipelineOptions()
    
    # CRITICAL: Enable table structure detection
    pipeline_options.do_table_structure = True
    
    # Table structure options - IMPROVED SETTINGS
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    
    # OCR options for better text extraction
    pipeline_options.do_ocr = False
    pipeline_options.ocr_options.force_full_page_ocr = False
    
    try:
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        logger.info("Docling converter initialized successfully")
    except Exception as conv_error:
        logger.error(f"Failed to initialize Docling converter: {conv_error}")
        logger.error(f"Cache status: {cache_available}")
        if not cache_available:
            logger.error("This error is likely due to missing model cache")
        raise

    # IMPROVED text splitter with table-aware separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=[
            "\n\n\n\n",
            "\n\n\n",
            "\n\n",
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n|",
            "|\n",
            "\n- ",
            "\n* ",
            ". ",
            ".\n",
            " ",
        ],
        keep_separator=True,
        length_function=len,
    )

    # Validate essential environment variables
    required_vars = [
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", 
        "PINECONE_API_KEY", "PINECONE_INDEX_NAME",
        "AWS_S3_BUCKET_NAME", "NEXTJS_CALLBACK_URL", "PYTHON_SERVICE_SECRET"
    ]
    if not all(os.getenv(var) for var in required_vars):
        logger.critical("CRITICAL: Missing one or more required environment variables!")

except Exception as e:
    logger.critical(f"CRITICAL: Failed to initialize clients: {e}")
    logger.critical(traceback.format_exc())

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    s3_key: str
    document_id: str

def extract_tables_from_docling(document: DoclingDocument) -> List[Dict]:
    """Extract tables with their context from Docling document"""
    tables = []
    
    try:
        # Get tables from document
        for table in document.tables:
            table_data = {
                "content": table.export_to_markdown(doc=document),
                "caption": getattr(table, 'caption', ''),
                "bbox": getattr(table, 'bbox', None),
                "page": getattr(table, 'page', None)
            }
            tables.append(table_data)
            logger.info(f"Extracted table: {table_data['content'][:100]}...")
    except Exception as e:
        logger.warning(f"Error extracting tables: {e}")
    
    return tables

def create_smart_chunks(markdown_content: str, tables: List[Dict]) -> List[Dict]:
    """Create intelligent chunks that preserve table structure"""
    chunks = []
    
    # First, identify table positions in markdown
    table_positions = []
    for i, table in enumerate(tables):
        table_content = table["content"]
        start_pos = markdown_content.find(table_content)
        if start_pos != -1:
            table_positions.append({
                "start": start_pos,
                "end": start_pos + len(table_content),
                "table_index": i,
                "content": table_content,
                "caption": table.get("caption", "")
            })
    
    # Sort by position
    table_positions.sort(key=lambda x: x["start"])
    
    current_pos = 0
    chunk_index = 0
    
    for table_pos in table_positions:
        # Add text before table as separate chunks
        text_before = markdown_content[current_pos:table_pos["start"]].strip()
        if text_before:
            text_chunks = text_splitter.split_text(text_before)
            for text_chunk in text_chunks:
                if len(text_chunk.strip()) > 20:
                    chunks.append({
                        "content": text_chunk,
                        "type": "text",
                        "chunk_index": chunk_index,
                        "contains_table": False
                    })
                    chunk_index += 1
        
        # Add table as a complete chunk with context
        table_chunk_content = table_pos["content"]
        
        # Add some context before and after the table
        context_before_start = max(0, table_pos["start"] - 150)
        context_after_end = min(len(markdown_content), table_pos["end"] + 150)
        
        context_before = markdown_content[context_before_start:table_pos["start"]].strip()
        context_after = markdown_content[table_pos["end"]:context_after_end].strip()
        
        # Create enhanced table chunk
        enhanced_table_content = ""
        if context_before:
            context_lines = context_before.split('\n')
            relevant_context = '\n'.join(context_lines[-3:]) if len(context_lines) > 3 else context_before
            enhanced_table_content += f"{relevant_context}\n\n"
        
        enhanced_table_content += table_chunk_content
        
        if context_after:
            context_lines = context_after.split('\n')
            relevant_context = '\n'.join(context_lines[:3]) if len(context_lines) > 3 else context_after
            enhanced_table_content += f"\n\n{relevant_context}"
        
        chunks.append({
            "content": enhanced_table_content,
            "type": "table",
            "chunk_index": chunk_index,
            "contains_table": True,
            "table_caption": table_pos["caption"]
        })
        chunk_index += 1
        
        current_pos = table_pos["end"]
    
    # Add remaining text after last table
    remaining_text = markdown_content[current_pos:].strip()
    if remaining_text:
        text_chunks = text_splitter.split_text(remaining_text)
        for text_chunk in text_chunks:
            if len(text_chunk.strip()) > 20:
                chunks.append({
                    "content": text_chunk,
                    "type": "text",
                    "chunk_index": chunk_index,
                    "contains_table": False
                })
                chunk_index += 1
    
    return chunks

def improve_chunk_quality(chunks: List[Dict]) -> List[Dict]:
    """Filter and improve chunk quality"""
    improved_chunks = []
    
    for chunk in chunks:
        content = chunk["content"].strip()
        
        # Skip very short chunks
        if len(content) < 30:
            continue
        
        # Skip metadata-only chunks
        if is_metadata_chunk(content):
            continue
        
        # Clean up content
        content = clean_chunk_content(content)
        
        if len(content.strip()) > 30:
            chunk["content"] = content
            improved_chunks.append(chunk)
    
    return improved_chunks

def is_metadata_chunk(content: str) -> bool:
    """Identify chunks that are primarily metadata"""
    content_lower = content.lower()
    
    # Header/footer patterns
    metadata_patterns = [
        r'página \d+ de \d+',
        r'boletín oficial del estado',
        r'núm\. \d+.*miércoles.*\d+ de.*\d+',
        r'sec\. iii\. pág\. \d+',
        r'cve: boe-a-\d+-\d+',
        r'verificable en https://www\.boe\.es'
    ]
    
    for pattern in metadata_patterns:
        if re.search(pattern, content_lower) and len(content) < 200:
            return True
    
    return False

def clean_chunk_content(content: str) -> str:
    """Clean and normalize chunk content"""
    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    # Remove image placeholders
    content = re.sub(r'<!-- image -->', '', content)
    content = re.sub(r'!\[image\].*?\n', '', content)
    
    # Clean up table formatting
    content = re.sub(r'\|\s*\|\s*\|', '||', content)
    
    return content.strip()

def is_image_file(file_path: str) -> bool:
    """Check if file is an image based on extension"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    _, ext = os.path.splitext(file_path.lower())
    return ext in image_extensions

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def process_image_with_openai(image_path: str) -> str:
    """Process image using OpenAI Vision API"""
    try:
        # Get image extension to determine MIME type
        _, ext = os.path.splitext(image_path.lower())
        mime_type = "image/jpeg" if ext in ['.jpg', '.jpeg'] else f"image/{ext[1:]}"
        
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        
        # Create the vision request
        response = openai_client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this image and extract all text content, data, and meaningful information. 
                            Structure your response as follows:
                            
                            ## Image Description
                            [Describe what you see in the image]
                            
                            ## Extracted Text
                            [All text found in the image, maintaining original formatting where possible]
                            
                            ## Data and Information
                            [Any structured data, tables, charts, diagrams, or other meaningful information]
                            
                            ## Context and Analysis
                            [Any additional context or analysis that would be helpful for document search and retrieval]
                            
                            Please be thorough and accurate in your extraction."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        logger.info(f"OpenAI Vision processing successful, extracted {len(content)} characters")
        return content
        
    except Exception as e:
        logger.error(f"OpenAI Vision processing failed: {e}")
        raise Exception(f"Image processing failed: {str(e)}")

async def notify_nextjs(document_id: str, success: bool, error_message: Optional[str] = None, max_retries: int = 3):
    """Sends completion status back to the Next.js API route with retry logic."""
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
        "X-Processing-Secret": PYTHON_SERVICE_SECRET
    }
    
    logger.info(f"Attempting to notify Next.js at {NEXTJS_CALLBACK_URL} for doc {document_id}, success={success}")

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(NEXTJS_CALLBACK_URL, json=payload, headers=headers)
                response.raise_for_status()
                logger.info(f"Successfully notified Next.js for document {document_id}. Status: {success}.")
                return
                
        except httpx.TimeoutException as e:
            logger.warning(f"Timeout error notifying Next.js for document {document_id} (attempt {attempt + 1}/{max_retries}): {e}")
        except httpx.ConnectError as e:
            logger.warning(f"Connection error notifying Next.js for document {document_id} (attempt {attempt + 1}/{max_retries}): {e}")
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP {e.response.status_code} error notifying Next.js for document {document_id} (attempt {attempt + 1}/{max_retries}): {e}")
        except Exception as e:
            logger.warning(f"Unexpected error notifying Next.js for document {document_id} (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
        
        # Wait before retry (exponential backoff)
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            logger.info(f"Retrying notification for document {document_id} in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
    
    logger.error(f"Failed to notify Next.js for document {document_id} after {max_retries} attempts")

@app.get("/health")
async def health_check():
    cache_status = verify_model_cache()
    return {
        "status": "healthy", 
        "models_cached": cache_status,
        "hf_offline": os.environ.get('HF_HUB_OFFLINE', 'not_set'),
        "cache_path": "/app/.cache/huggingface"
    }

@app.post("/process")
async def process_document(payload: ProcessRequest):
    """Enhanced document processing with better table handling"""
    s3_key = payload.s3_key
    document_id = payload.document_id
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
                if error_code == '404':
                    error_message = f"S3 Key not found: {s3_key}"
                    logger.error(error_message)
                    await notify_nextjs(document_id, success=False, error_message=error_message)
                    raise HTTPException(status_code=404, detail=error_message)
                else:
                    logger.error(error_message)
                    await notify_nextjs(document_id, success=False, error_message=error_message)
                    raise HTTPException(status_code=500, detail=error_message)

        # Extract folder_id from s3_key structure
        key_parts = s3_key.split('/')
        folder_id = key_parts[0] if len(key_parts) > 1 and key_parts[0] != 'root' else "root"

        # 2. Check if it's an image file and process accordingly
        if is_image_file(temp_path):
            logger.info(f"Processing IMAGE file with OpenAI Vision: {temp_path} for doc: {document_id}")
            
            try:
                # Process image with OpenAI Vision
                extracted_content = await process_image_with_openai(temp_path)
                logger.info(f"Image processing successful, extracted {len(extracted_content)} characters")

                # Create chunks from the extracted content
                chunks_data = []
                if extracted_content and len(extracted_content.strip()) > 30:
                    # Split the content into manageable chunks
                    content_chunks = text_splitter.split_text(extracted_content)
                    
                    for i, chunk_content in enumerate(content_chunks):
                        if len(chunk_content.strip()) > 30:
                            chunks_data.append({
                                "content": chunk_content,
                                "type": "image_extracted",
                                "chunk_index": i,
                                "contains_table": False
                            })

                logger.info(f"Created {len(chunks_data)} chunks from image content")

            except Exception as vision_error:
                logger.error(f"Image processing failed for doc {document_id}: {vision_error}")
                await notify_nextjs(document_id, success=False, 
                                  error_message=f"Image processing failed: {vision_error}")
                raise HTTPException(status_code=500, detail=f"Image processing failed: {vision_error}")
        
        else:
            # 3. ENHANCED Docling Processing for non-image files
            logger.info(f"Processing DOCUMENT file with ENHANCED Docling: {temp_path} for doc: {document_id}")
            
            try:
                # Convert document with Docling with timeout protection
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(2400)  # 40 minute timeout
                
                logger.info("Converting document with Docling...")
                result = converter.convert(temp_path)
                signal.alarm(0)  # Cancel timeout
                logger.info("Document conversion successful.")

                # Extract tables BEFORE converting to markdown
                logger.info("Extracting tables from Docling document...")
                tables = extract_tables_from_docling(result.document)
                logger.info(f"Extracted {len(tables)} tables from document")

                # Export to markdown
                logger.info("Exporting document to markdown...")
                markdown = result.document.export_to_markdown()
                logger.info(f"Markdown export successful, length: {len(markdown)} characters")

                # Create smart chunks that preserve table structure
                logger.info("Creating smart chunks with table preservation...")
                chunks_data = create_smart_chunks(markdown, tables)
                logger.info(f"Created {len(chunks_data)} initial chunks")

                # Improve chunk quality
                logger.info("Improving chunk quality...")
                chunks_data = improve_chunk_quality(chunks_data)
                logger.info(f"Final chunk count after quality improvement: {len(chunks_data)}")

            except TimeoutError:
                signal.alarm(0)  # Cancel any pending alarm
                logger.error(f"Docling conversion timed out after 20 minutes for doc {document_id}")
                await notify_nextjs(document_id, success=False, 
                                  error_message="Document processing timed out after 20 minutes")
                raise HTTPException(status_code=500, detail="Document processing timed out after 20 minutes")
            except Exception as docling_error:
                logger.error(f"Docling processing failed for doc {document_id}: {docling_error}")
                logger.error(traceback.format_exc())
                await notify_nextjs(document_id, success=False, 
                                  error_message=f"Document processing failed: {docling_error}")
                raise HTTPException(status_code=500, detail=f"Document processing failed: {docling_error}")

        # Check if we have any chunks (common for both image and document processing)
        if not chunks_data:
            logger.warning(f"No chunks generated for doc: {document_id}.")
            await notify_nextjs(document_id, success=True, 
                              error_message="Document processed but contained no text chunks.")
            return {"message": f"Document {document_id} processed, no text chunks found."}

        # Log processing statistics
        file_type = "image" if is_image_file(temp_path) else "document"
        table_chunks = [c for c in chunks_data if c.get("contains_table", False)]
        text_chunks = [c for c in chunks_data if not c.get("contains_table", False)]
        
        logger.info(f"Processing statistics for {file_type}:")
        logger.info(f"  - Total chunks: {len(chunks_data)}")
        logger.info(f"  - Table chunks: {len(table_chunks)}")
        logger.info(f"  - Text chunks: {len(text_chunks)}")
        
        if table_chunks and file_type == "document":
            logger.info(f"  - Table chunks content preview:")
            for i, chunk in enumerate(table_chunks[:2]):
                logger.info(f"    Table {i+1}: {chunk['content'][:200]}...")

        # 3. Prepare records for Pinecone
        records = []
        for chunk_data in chunks_data:
            record_id = f"{document_id}_chunk_{chunk_data['chunk_index']}"
            
            # Prepare metadata for the chunk
            metadata = {
                "documentId": document_id,
                "folderId": folder_id,
                "originalFilename": original_filename,
                "chunkIndex": chunk_data['chunk_index'],
                "chunkLength": len(chunk_data['content']),
                "chunkType": chunk_data['type'],
                "containsTable": chunk_data.get('contains_table', False)
            }
            
            # Add table-specific metadata
            if chunk_data.get('contains_table'):
                metadata["tableCaption"] = chunk_data.get('table_caption', '')
            
            # Add image-specific metadata
            if chunk_data['type'] == 'image_extracted':
                metadata["isImageDerived"] = True
                metadata["imageS3Key"] = s3_key
                metadata["imageS3Bucket"] = S3_BUCKET_NAME

            # Create record for Pinecone
            record = {
                "_id": record_id,
                "text": chunk_data['content'],
                **metadata
            }
            records.append(record)

        # 4. Upsert records to Pinecone
        batch_size = 96
        logger.info(f"Upserting {len(records)} records to Pinecone namespace '{PINECONE_NAMESPACE}' for doc {document_id}...")
        
        try:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                pinecone_index.upsert_records(
                    namespace=PINECONE_NAMESPACE,
                    records=batch
                )
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size} for doc {document_id}")
            
            logger.info(f"Successfully upserted all records for document {document_id}")

            # 5. Notify Next.js of success
            logger.info(f"About to notify Next.js of SUCCESS for individual document: {document_id}")
            await notify_nextjs(document_id, success=True)
            logger.info(f"Finished notifying Next.js of SUCCESS for individual document: {document_id}")

            return {
                "message": f"Processing complete for {document_id}",
                "file_type": file_type,
                "chunks_created": len(records),
                "table_chunks": len([r for r in records if r.get("containsTable", False)]),
                "text_chunks": len([r for r in records if not r.get("containsTable", False)]),
                "image_chunks": len([r for r in records if r.get("chunkType") == "image_extracted"])
            }

        except Exception as pinecone_error:
            logger.error(f"Pinecone upsert failed for doc {document_id}: {pinecone_error}")
            await notify_nextjs(document_id, success=False, 
                              error_message=f"Pinecone upsert failed: {pinecone_error}")
            raise HTTPException(status_code=500, detail=f"Pinecone upsert failed: {pinecone_error}")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        error_message = f"Unexpected error processing document {document_id}: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        await notify_nextjs(document_id, success=False, error_message=error_message)
        raise HTTPException(status_code=500, detail=error_message)
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Temporary file {temp_path} removed for doc {document_id}")
            except Exception as cleanup_error:
                 logger.error(f"Error removing temp file {temp_path} for doc {document_id}: {cleanup_error}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Uvicorn server on 0.0.0.0:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)