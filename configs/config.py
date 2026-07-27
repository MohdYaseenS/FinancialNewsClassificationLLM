import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
DATASET_NAME = os.getenv("DATASET_NAME")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
USE_QLORA = os.getenv("USE_QLORA") == "true"

MODEL_REGISTRY = {
    "qwen": "Qwen/Qwen3-0.6B",
    "gemma": "google/gemma-2b",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}

FRONTEND_PORT = os.getenv("FRONTEND_PORT")
BACKTEND_PORT = os.getenv("BACKEND_PORT")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")