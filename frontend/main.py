"""
med-EZ Backend — FastAPI Application
Connects ui1.html (landing), ui2.html (prescription reader), u3.html (AI chatbot)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os

from ocr import extract_prescription
from explanation import explain_medicines
from chatbot import chat_with_doctor
from drug_interactions import check_interactions
from symptom_checker import check_symptoms

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="med-EZ API",
    description="Healthcare AI backend — prescription OCR, medicine explanations, AI chatbot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the three HTML pages as static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ─────────────────────────────────────────────
# Page routes  (serve HTML files directly)
# ─────────────────────────────────────────────
@app.get("/")
def landing():
    return FileResponse("frontend/ui1.html")

@app.get("/prescriptions")
def prescriptions():
    return FileResponse("frontend/ui2.html")

@app.get("/chatbot")
def chatbot_page():
    return FileResponse("frontend/u3.html")


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str           # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    context: Optional[str] = None   # optional prescription context

class ExplainRequest(BaseModel):
    medicines: list[dict]

class InteractionRequest(BaseModel):
    medicine_names: list[str]

class SymptomRequest(BaseModel):
    symptoms: str
    age: Optional[int] = None
    gender: Optional[str] = None


# ─────────────────────────────────────────────
# API — Prescription OCR  (ui2.html)
# ─────────────────────────────────────────────
@app.post("/api/prescription/extract")
async def extract(file: UploadFile = File(...)):
    """
    Upload a prescription image → get structured JSON back.
    Used by: ui2.html  →  Image Upload button
    """
    allowed = {"image/jpeg", "image/png", "image/webp", "image/gif", "application/pdf"}
    if file.content_type not in allowed:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Upload JPG, PNG, WEBP or PDF.")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:   # 10 MB limit
        raise HTTPException(413, "File too large. Maximum size is 10 MB.")

    result = await extract_prescription(image_bytes, file.content_type)

    if "error" in result:
        raise HTTPException(422, result["error"])

    return result


# ─────────────────────────────────────────────
# API — Medicine explanations  (ui2.html)
# ─────────────────────────────────────────────
@app.post("/api/prescription/explain")
async def explain(req: ExplainRequest):
    """
    Take the extracted medicines list → return plain-language explanations.
    Used by: ui2.html  →  after OCR, to populate dosage/frequency/duration cards
    """
    if not req.medicines:
        raise HTTPException(400, "No medicines provided.")

    explanations = await explain_medicines(req.medicines)
    return {"explanations": explanations}


# ─────────────────────────────────────────────
# API — AI Chatbot  (u3.html)
# ─────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Multi-turn chat with the AI doctor.
    Used by: u3.html  →  message input box
    Optional: pass prescription context so the AI knows what medicines the patient has.
    """
    if not req.messages:
        raise HTTPException(400, "No messages provided.")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    reply = await chat_with_doctor(messages, context=req.context)
    return {"reply": reply}


# ─────────────────────────────────────────────
# API — Drug interactions  (ui2.html side panel)
# ─────────────────────────────────────────────
@app.post("/api/interactions")
async def interactions(req: InteractionRequest):
    """
    Check for interactions between a list of drug names.
    Used by: ui2.html  →  Drug Interactions button
    """
    if len(req.medicine_names) < 2:
        raise HTTPException(400, "Provide at least 2 medicine names to check interactions.")

    result = await check_interactions(req.medicine_names)
    return result


# ─────────────────────────────────────────────
# API — Symptom checker  (ui2.html side panel / u3.html)
# ─────────────────────────────────────────────
@app.post("/api/symptoms")
async def symptoms(req: SymptomRequest):
    """
    Analyse patient symptoms and return possible causes + recommendations.
    Used by: ui2.html  →  Symptom Checker button
             u3.html   →  chatbot can call this internally
    """
    if not req.symptoms.strip():
        raise HTTPException(400, "Symptoms text cannot be empty.")

    result = await check_symptoms(req.symptoms, req.age, req.gender)
    return result


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "med-EZ API v1.0"}
