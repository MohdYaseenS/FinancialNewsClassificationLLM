from fastapi import FastAPI
from pydantic import BaseModel

from backend.inference import generate

app = FastAPI()


class GenerateRequest(BaseModel):
    text: str


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/generate")
def generate_text(request: GenerateRequest):
    response = generate(request.text)

    return {"response": response}