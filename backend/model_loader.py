from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

from configs.config import MODEL_NAME, MODEL_REGISTRY, USE_QLORA


def load_model():

    model_id = MODEL_REGISTRY[MODEL_NAME]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if USE_QLORA:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

    else:

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto"
        )

    return model, tokenizer