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

# Server configuration
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))

FRONTEND_HOST = os.getenv("FRONTEND_HOST", "127.0.0.1")
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 7860))

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")