from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.config import MODEL_NAME, MODEL_REGISTRY

def load_model():

    model_id = MODEL_REGISTRY[MODEL_NAME]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )

    return model, tokenizer