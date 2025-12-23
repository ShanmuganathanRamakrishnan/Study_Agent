"""
FastAPI server for Study Agent
Exposes the API contract endpoints wrapping the frozen RAG core.
"""
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sys

# Import the frozen core
from study_agent import RAGSystem, load_model, generate_questions, generate_solution, parse_questions

# Import PDF extractor
from pdf_extractor import extract_text_from_pdf, validate_pdf_content

# Import bounded math support
from math_support import should_attempt_math, classify_math_query, solve_simple_math

# Import intent router
from intent_router import classify_intent, get_intent_response, log_intent

# Import conversation context
from conversation_context import (
    get_buffer, reset_buffer, is_follow_up_query, expand_follow_up_query
)

# --- Pydantic Models ---

class AskQuestionRequest(BaseModel):
    query: str

class AskQuestionResponse(BaseModel):
    """
    Strict API response contract for ask_question endpoint.
    All fields are structured - no markdown in any field.
    """
    status: str  # "ANSWERED" | "REFUSED" | "ERROR"
    confidence: str = "LOW"  # "HIGH" | "LOW"
    domain: str = ""  # e.g., "THERMODYNAMICS", "BIOLOGY"
    answer: Optional[str] = None  # Plain text answer (no markdown)
    quote: Optional[str] = None  # Exact supporting quote or null
    reason: Optional[str] = None  # Reason for refusal (only if REFUSED)


def parse_model_response(raw_output: str) -> dict:
    """
    Parse raw LLM output into structured fields.
    Removes markdown artifacts like **Answer:** and **Quote:**
    
    Returns: { "answer": str, "quote": str | None }
    """
    answer = raw_output
    quote = None
    
    # Pattern 1: **Answer:** ... **Quote:** "..."
    import re
    answer_match = re.search(r'\*\*Answer:\*\*\s*(.+?)(?=\*\*Quote:|$)', raw_output, re.DOTALL)
    quote_match = re.search(r'\*\*Quote:\*\*\s*"?([^"]+)"?', raw_output, re.DOTALL)
    
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Pattern 2: Answer: ... Quote: "..."
        simple_answer_match = re.search(r'Answer:\s*(.+?)(?=Quote:|$)', raw_output, re.DOTALL)
        if simple_answer_match:
            answer = simple_answer_match.group(1).strip()
    
    if quote_match:
        quote = quote_match.group(1).strip()
    else:
        simple_quote_match = re.search(r'Quote:\s*"?([^"]+)"?', raw_output, re.DOTALL)
        if simple_quote_match:
            quote = simple_quote_match.group(1).strip()
    
    # Clean up any remaining markdown bold markers
    answer = re.sub(r'\*\*', '', answer).strip()
    if quote:
        quote = re.sub(r'\*\*', '', quote).strip()
    
    # If no structured format found, just clean markdown from raw content
    if not answer_match and 'Answer:' not in raw_output:
        answer = re.sub(r'\*\*', '', raw_output).strip()
    
    return {"answer": answer, "quote": quote}

class GenerateStudyGuideRequest(BaseModel):
    topic: str
    difficulty: str  # "easy" | "medium" | "hard"

class QuestionItem(BaseModel):
    question: str
    expected_answer_length: str

class GenerateStudyGuideResponse(BaseModel):
    status: str  # "generated" | "failed" | "error"
    questions: List[QuestionItem] = []
    topic: str = ""
    difficulty: str = ""
    count: int = 0
    failure_reason: Optional[str] = None

class UploadMaterialRequest(BaseModel):
    content: str
    domain: str

class UploadMaterialResponse(BaseModel):
    status: str  # "indexed" | "rejected" | "error"
    domain: str = ""
    chunks_created: int = 0
    word_count: int = 0
    rejection_reason: Optional[str] = None

class SystemStatusResponse(BaseModel):
    status: str  # "ready" | "loading" | "error"
    indexed_domains: List[str] = []
    total_chunks: int = 0
    model_loaded: bool = False
    embedding_model: str = ""
    generation_model: str = ""

# --- Initialize App ---

app = FastAPI(
    title="Study Agent API",
    description="Educational Q&A system with RAG",
    version="1.0.0"
)

# CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
rag_system: Optional[RAGSystem] = None
tokenizer = None
model = None
user_documents = {}  # Track user-uploaded documents

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system and models on startup."""
    global rag_system, tokenizer, model
    
    print("Initializing Study Agent API...")
    
    # Initialize RAG
    rag_system = RAGSystem()
    
    # Index the sample documents from study_agent.py
    from study_agent import thermo_text, info_theory_text, bio_text, os_text, stats_text
    documents = {
        "THERMODYNAMICS": thermo_text,
        "INFORMATION_THEORY": info_theory_text,
        "BIOLOGY": bio_text,
        "OPERATING_SYSTEMS": os_text,
        "STATISTICS": stats_text,
    }
    rag_system.index_documents(documents)
    
    # Load generation model
    print("Loading generation model...")
    tokenizer, model = load_model()
    
    print("Study Agent API ready!")

# --- Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker/K8s probes."""
    return {
        "status": "healthy",
        "rag_ready": rag_system is not None,
        "model_ready": model is not None
    }

@app.get("/system_status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get current system state."""
    if rag_system is None:
        return SystemStatusResponse(
            status="loading",
            indexed_domains=[],
            total_chunks=0,
            model_loaded=False,
            embedding_model="",
            generation_model=""
        )
    
    # Extract unique domains from chunks
    domains = set()
    for chunk in rag_system.chunks:
        if chunk.startswith("[") and "]" in chunk:
            domain = chunk.split("]")[0][1:]
            domains.add(domain)
    
    return SystemStatusResponse(
        status="ready" if model is not None else "loading",
        indexed_domains=sorted(list(domains)),
        total_chunks=len(rag_system.chunks),
        model_loaded=model is not None,
        embedding_model="all-MiniLM-L6-v2",
        generation_model="Qwen/Qwen3-1.7B" if model else ""
    )

@app.post("/ask_question", response_model=AskQuestionResponse)
async def ask_question(request: AskQuestionRequest):
    """
    Ask a question about indexed study materials.
    Returns structured JSON with no markdown in any field.
    """
    if rag_system is None or model is None:
        return AskQuestionResponse(
            status="ERROR",
            reason="System not ready"
        )
    
    query = request.query.strip()
    if not query:
        return AskQuestionResponse(
            status="REFUSED",
            reason="Empty query"
        )
    
    # --- Intent Routing (preprocessing layer) ---
    intent_result = classify_intent(query)
    log_intent(query, intent_result)
    
    # Handle non-QUESTION intents before RAG
    if intent_result.intent == "GREETING":
        return AskQuestionResponse(
            status="ANSWERED",
            confidence="HIGH",
            domain="SYSTEM",
            answer="Ask a question about your uploaded materials.",
            quote=None
        )
    
    if intent_result.intent == "NOISE":
        return AskQuestionResponse(
            status="REFUSED",
            reason="Please enter a question about your study materials."
        )
    
    if intent_result.intent == "CLARIFICATION":
        return AskQuestionResponse(
            status="REFUSED",
            confidence="LOW",
            reason="Could you please rephrase or provide more detail? I work best with specific questions."
        )
    
    # STUDY_REQUEST and QUESTION proceed to existing flow
    
    # --- Bounded Math Support ---
    # Check if query looks like math and route accordingly
    if should_attempt_math(query):
        classification, _ = classify_math_query(query)
        
        # Complex math: refuse explicitly
        if classification == "COMPLEX_REFUSED":
            return AskQuestionResponse(
                status="REFUSED",
                confidence="LOW",
                reason="This math problem exceeds the scope of simple arithmetic. I can only help with basic single-step calculations (e.g., 5+3, 20% of 100)."
            )
        
        # Simple math: compute directly
        if classification in ["SIMPLE_ARITHMETIC", "SIMPLE_PERCENTAGE"]:
            solved, result, explanation = solve_simple_math(query)
            if solved and result:
                return AskQuestionResponse(
                    status="ANSWERED",
                    confidence="HIGH",
                    domain="MATH",
                    answer=f"The answer is {result}.",
                    quote=explanation
                )
    
    # --- Conversation Context ---
    conv_buffer = get_buffer()
    
    # Detect and expand follow-up queries
    retrieval_query = query
    if is_follow_up_query(query, conv_buffer):
        retrieval_query = expand_follow_up_query(query, conv_buffer)
        print(f"[CONTEXT] Follow-up detected, using expanded query for retrieval", flush=True)
    
    # Add current user query to buffer AFTER checking for follow-up
    conv_buffer.add_user_query(query)
    
    # --- Regular RAG Flow ---
    
    try:
        # Retrieve with confidence - use expanded query for follow-ups
        chunks, confidence = rag_system.retrieve_with_confidence(retrieval_query)
        
        if not chunks:
            return AskQuestionResponse(
                status="REFUSED",
                confidence="LOW",
                reason="No relevant content found"
            )
        
        # Extract domain from first chunk (format: "[DOMAIN] content...")
        context = chunks[0]
        domain = "UNKNOWN"
        if context.startswith("[") and "]" in context:
            domain = context.split("]")[0][1:]
        
        # Build enhanced context with conversation history
        conversation_context = conv_buffer.get_recent_context()
        enhanced_context = context
        if conversation_context and conv_buffer.has_context():
            # Include recent conversation for coherence
            enhanced_context = f"{conversation_context}\n\n[Retrieved Content]\n{context}"
        
        # Generate answer using the tutor (with enhanced context)
        raw_answer = generate_solution(tokenizer, model, enhanced_context, query, confidence)
        
        # Check for refusal in answer
        if "The provided text does not contain this information" in raw_answer:
            return AskQuestionResponse(
                status="REFUSED",
                confidence=confidence,
                domain=domain,
                reason="The provided text does not contain this information"
            )
        
        # Parse raw model output into structured fields
        parsed = parse_model_response(raw_answer)
        
        # Add assistant response to buffer
        conv_buffer.add_assistant_response(parsed["answer"], domain=domain)
        
        return AskQuestionResponse(
            status="ANSWERED",
            confidence=confidence,
            domain=domain,
            answer=parsed["answer"],
            quote=parsed["quote"]
        )
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        return AskQuestionResponse(
            status="ERROR",
            reason=str(e)
        )

@app.post("/generate_study_guide", response_model=GenerateStudyGuideResponse)
async def generate_study_guide(request: GenerateStudyGuideRequest):
    """Generate exam-style questions for a topic."""
    if rag_system is None or model is None:
        return GenerateStudyGuideResponse(
            status="error",
            failure_reason="System not ready"
        )
    
    topic = request.topic.strip().upper()
    difficulty = request.difficulty.lower()
    
    if difficulty not in ["easy", "medium", "hard"]:
        return GenerateStudyGuideResponse(
            status="failed",
            topic=topic,
            difficulty=difficulty,
            failure_reason="Invalid difficulty level"
        )
    
    # Find relevant chunks for the topic
    topic_chunks = [c for c in rag_system.chunks if topic in c.upper()]
    
    if not topic_chunks:
        return GenerateStudyGuideResponse(
            status="failed",
            topic=topic,
            difficulty=difficulty,
            failure_reason="Topic not found in indexed materials"
        )
    
    # Generate questions using the examiner
    context = " ".join(topic_chunks[:3])  # Use up to 3 chunks
    
    try:
        questions_text = generate_questions(tokenizer, model, context)
        questions = parse_questions(questions_text)
        
        # Map difficulty to expected answer length
        answer_lengths = {
            "easy": "8-12 words",
            "medium": "12-18 words",
            "hard": "15-25 words"
        }
        
        question_items = [
            QuestionItem(question=q, expected_answer_length=answer_lengths[difficulty])
            for q in questions[:5]  # Limit to 5 questions
        ]
        
        return GenerateStudyGuideResponse(
            status="generated",
            questions=question_items,
            topic=topic,
            difficulty=difficulty,
            count=len(question_items)
        )
    except Exception as e:
        print(f"Generation error: {e}")
        return GenerateStudyGuideResponse(
            status="failed",
            topic=topic,
            difficulty=difficulty,
            failure_reason=str(e)
        )

@app.post("/upload_material", response_model=UploadMaterialResponse)
async def upload_material(request: UploadMaterialRequest):
    """Index new study material."""
    global user_documents
    
    if rag_system is None:
        return UploadMaterialResponse(
            status="error",
            rejection_reason="System not ready"
        )
    
    content = request.content.strip()
    domain = request.domain.strip().upper()
    
    if not content:
        return UploadMaterialResponse(
            status="rejected",
            domain=domain,
            rejection_reason="Content is empty"
        )
    
    if not domain:
        return UploadMaterialResponse(
            status="rejected",
            rejection_reason="Domain tag is required"
        )
    
    # Count words
    word_count = len(content.split())
    
    if word_count < 100:
        return UploadMaterialResponse(
            status="rejected",
            domain=domain,
            word_count=word_count,
            rejection_reason="Content too short (minimum 100 words)"
        )
    
    # Index the new material incrementally (NOT re-indexing all documents)
    try:
        import time
        start_time = time.time()
        
        # Store user document for persistence
        user_documents[domain] = content
        
        print(f"[UPLOAD_MATERIAL] Incrementally indexing {domain}...", flush=True)
        
        # Use incremental indexing - only embeds new chunks
        chunks_created = rag_system.add_document(domain, content)
        
        index_time = time.time() - start_time
        print(f"[UPLOAD_MATERIAL] Indexed {word_count} words as {domain}, created {chunks_created} chunks in {index_time:.1f}s", flush=True)
        
        return UploadMaterialResponse(
            status="indexed",
            domain=domain,
            chunks_created=chunks_created,
            word_count=word_count
        )
    except Exception as e:
        print(f"[UPLOAD_MATERIAL] Indexing error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return UploadMaterialResponse(
            status="error",
            domain=domain,
            rejection_reason=str(e)
        )

@app.post("/upload_file", response_model=UploadMaterialResponse)
async def upload_file(
    file: UploadFile = File(...),
    domain: str = Form(...)
):
    """
    Upload a file (PDF or TXT) for indexing.
    PDFs are text-extracted only - no OCR, no image processing.
    Same chunking/embedding pipeline as text content.
    
    LIMITATIONS:
    - No equation solving or symbolic math
    - Scanned PDFs will return empty text
    - Images and diagrams are ignored
    """
    global user_documents
    
    if rag_system is None:
        return UploadMaterialResponse(
            status="error",
            rejection_reason="System not ready"
        )
    
    domain = domain.strip().upper()
    if not domain:
        return UploadMaterialResponse(
            status="rejected",
            rejection_reason="Domain tag is required"
        )
    
    # Read file content
    try:
        file_bytes = await file.read()
    except Exception as e:
        return UploadMaterialResponse(
            status="error",
            domain=domain,
            rejection_reason=f"Failed to read file: {str(e)}"
        )
    
    filename = file.filename or ""
    file_ext = filename.lower().split(".")[-1] if "." in filename else ""
    
    # Extract text based on file type
    content = ""
    if file_ext == "pdf":
        try:
            content, metadata = extract_text_from_pdf(file_bytes)
            print(f"PDF extracted: {metadata['word_count']} words, {metadata['page_count']} pages")
            
            # Validate PDF content
            is_valid, reason = validate_pdf_content(content)
            if not is_valid:
                return UploadMaterialResponse(
                    status="rejected",
                    domain=domain,
                    word_count=len(content.split()) if content else 0,
                    rejection_reason=reason
                )
        except ValueError as e:
            return UploadMaterialResponse(
                status="rejected",
                domain=domain,
                rejection_reason=str(e)
            )
    elif file_ext == "txt":
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content = file_bytes.decode("latin-1")
            except:
                return UploadMaterialResponse(
                    status="rejected",
                    domain=domain,
                    rejection_reason="Failed to decode text file"
                )
    else:
        return UploadMaterialResponse(
            status="rejected",
            domain=domain,
            rejection_reason=f"Unsupported file type: .{file_ext}. Only .pdf and .txt are supported."
        )
    
    content = content.strip()
    if not content:
        return UploadMaterialResponse(
            status="rejected",
            domain=domain,
            rejection_reason="File content is empty"
        )
    
    # Count words
    word_count = len(content.split())
    print(f"[UPLOAD] Content extracted: {word_count} words", flush=True)
    
    if word_count < 100:
        return UploadMaterialResponse(
            status="rejected",
            domain=domain,
            word_count=word_count,
            rejection_reason="Content too short (minimum 100 words)"
        )
    
    # Index the material incrementally (NOT re-indexing all documents)
    try:
        import time
        start_time = time.time()
        
        # Store user document for persistence
        user_documents[domain] = content
        
        print(f"[UPLOAD_FILE] Incrementally indexing {domain} (from {file_ext.upper()})...", flush=True)
        
        # Use incremental indexing - only embeds new chunks
        chunks_created = rag_system.add_document(domain, content)
        
        index_time = time.time() - start_time
        print(f"[UPLOAD_FILE] Indexed {word_count} words as {domain}, created {chunks_created} chunks in {index_time:.1f}s", flush=True)
        
        return UploadMaterialResponse(
            status="indexed",
            domain=domain,
            chunks_created=chunks_created,
            word_count=word_count
        )
    except Exception as e:
        print(f"[UPLOAD_FILE] Indexing error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return UploadMaterialResponse(
            status="error",
            domain=domain,
            rejection_reason=str(e)
        )

# --- Health Check ---

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
