import torch

from backend.model_loader import load_model
from backend.utils import extract_pred_label
from configs.config import SYSTEM_PROMPT

model, tokenizer = load_model()


def generate(text):

    messages = [
        {
            "role": "user",
            "content": f"{SYSTEM_PROMPT}\nSentence: {text}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

    prediction = tokenizer.decode(
        outputs[0],
        skip_special_tokens=False
    )

    label = extract_pred_label(prediction)

    return label