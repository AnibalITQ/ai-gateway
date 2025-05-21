from fastapi import FastAPI, UploadFile, File
import torchaudio
import io
from model import STTModel

app = FastAPI()
model = STTModel()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    content = await file.read()
    audio_array, _ = torchaudio.load(io.BytesIO(content))
    return model.transcribe(audio_array)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)