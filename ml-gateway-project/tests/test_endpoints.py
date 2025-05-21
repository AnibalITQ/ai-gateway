import pytest
from httpx import AsyncClient
import base64

# URL base cuando se ejecuta con docker-compose
BASE_URL = "http://localhost:8000"

@pytest.mark.asyncio
async def test_generate_endpoint():
    async with AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/generate",
            json={"prompt": "Write a short story about a robot:"}
        )
        assert response.status_code == 200
        assert "generated_text" in response.json()

@pytest.mark.asyncio
async def test_qa_endpoint():
    async with AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/qa",
            json={
                "question": "Who is the protagonist?",
                "context": "Alice went to wonderland and met a rabbit."
            }
        )
        assert response.status_code == 200
        assert "answer" in response.json()

@pytest.mark.asyncio
async def test_transcribe_endpoint():
    # Create a small test audio file or use a pre-recorded one
    async with AsyncClient() as client:
        with open("test_audio.wav", "rb") as f:
            response = await client.post(
                f"{BASE_URL}/transcribe",
                files={"file": ("test_audio.wav", f, "audio/wav")}
            )
        assert response.status_code == 200
        assert "transcription" in response.json()