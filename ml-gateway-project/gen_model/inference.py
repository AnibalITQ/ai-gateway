from fastapi import FastAPI
from pydantic import BaseModel
from model import TextGenerator

app = FastAPI()
model = TextGenerator()

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(request: GenerateRequest):
    return model.generate(request.prompt)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)