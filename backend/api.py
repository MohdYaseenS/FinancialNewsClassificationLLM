from fastapi import FastAPI
from backend.inference import generate

app = FastAPI()

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/generate")
def generate_text(prompt: str):

    response = generate(prompt)

    return {"response": response}