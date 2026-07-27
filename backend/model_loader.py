import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from configs.config import (
    MODEL_NAME,
    MODEL_REGISTRY,
    OUTPUT_DIR,
    USE_QLORA,
)

MODEL_DIR = os.path.join(OUTPUT_DIR, MODEL_NAME)

def load_model(trainable=False):
    """
    Loads the model for both training and inference.

    If a LoRA adapter exists in OUTPUT_DIR, it is loaded automatically.
    Otherwise, the base model is returned.

    Args:
        trainable (bool):
            True  -> LoRA adapter remains trainable (training)
            False -> LoRA adapter loaded for inference

    Returns:
        model, tokenizer
    """

    model_id = MODEL_REGISTRY[MODEL_NAME]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------
    # Load Base Model
    # -----------------------------

    if USE_QLORA:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )

    else:

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
        )

    # -----------------------------
    # Load LoRA Adapter (if present)
    # -----------------------------
    if os.path.isdir(MODEL_DIR):

        adapter_config = os.path.join(MODEL_DIR, "adapter_config.json")

        if os.path.isfile(adapter_config):

            print(f"Found LoRA adapter in '{MODEL_DIR}'. Loading fine-tuned model...")

            model = PeftModel.from_pretrained(
                model,
                MODEL_DIR,
                is_trainable=trainable,
            )

            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        else:
            print(f"No LoRA adapter found in '{MODEL_DIR}'. Using base model.")

    else:

        print(f"Model directory '{MODEL_DIR}' does not exist. Using base model.")

    return model, tokenizer