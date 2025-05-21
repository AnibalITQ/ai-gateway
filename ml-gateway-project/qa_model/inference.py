from fastapi import FastAPI
from pydantic import BaseModel
from model import QAModel

app = FastAPI()
model = QAModel()

class QuestionRequest(BaseModel):
    question: str
    context: str

@app.post("/predict")
async def predict(request: QuestionRequest):
    return model.predict(request.question, request.context)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)