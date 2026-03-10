from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from configs.config import MODEL_NAME, MODEL_REGISTRY
import torch

def load_model():

    model_id = MODEL_REGISTRY[MODEL_NAME]

    # Load Base Model (Quantized)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4", # Recommended type
        bnb_4bit_compute_dtype = torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config = quantization_config,
        torch_dtype = torch.float16,
        device_map = "auto",
    )

    return model, tokenizer