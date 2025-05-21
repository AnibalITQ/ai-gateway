from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import httpx
from ..core.config import settings

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str
    context: str

class GenerateRequest(BaseModel):
    prompt: str

@router.post("/qa")
async def question_answering(request: QuestionRequest):
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                f"{settings.MODEL_QA_URL}/predict",
                json=request.dict()
            )
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"QA service unavailable: {exc}")

@router.post("/generate")
async def text_generation(request: GenerateRequest):
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                f"{settings.MODEL_GEN_URL}/generate",
                json=request.dict()
            )
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Generation service unavailable: {exc}")

@router.post("/transcribe")
async def speech_to_text(file: UploadFile = File(...)):
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            files = {"file": ("temp", await file.read(), file.content_type)}
            response = await client.post(
                f"{settings.MODEL_STT_URL}/transcribe",
                files=files
            )
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"STT service unavailable: {exc}")