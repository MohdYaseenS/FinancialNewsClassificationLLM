from transformers import pipeline
from backend.model_loader import load_model

model, tokenizer = load_model()

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

def generate(prompt):

    output = generator(
        prompt,
        max_new_tokens=150,
        temperature=0.7
    )

    return output[0]["generated_text"]