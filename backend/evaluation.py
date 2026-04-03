import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from backend.dataset_loader import load_training_dataset
from configs.config import MODEL_NAME, MODEL_REGISTRY, OUTPUT_DIR


# ---------------------------------------------------
# Load trained model (Base + LoRA adapter)
# ---------------------------------------------------

def load_trained_model():

    model_id = MODEL_REGISTRY[MODEL_NAME]

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,   # force real weights
        device_map=None              # IMPORTANT: disable auto offloading
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        OUTPUT_DIR,
        is_trainable=False
    )

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------

def evaluate_model(model, tokenizer, test_dataset):

    print("\nRunning evaluation on test dataset...")

    y_true = []
    y_pred = []

    model.eval()

    for example in test_dataset:

        prompt = example["text"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )

        prediction = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        pred_label = prediction.split("\n")[-1].strip()
        true_label = example["text"].split("\n")[-1].strip()

        y_pred.append(pred_label)
        y_true.append(true_label)

    return y_true, y_pred


# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------

def save_confusion_matrix(y_true, y_pred):

    labels = ["positive", "negative", "neutral"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        df_cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    path = f"{OUTPUT_DIR}/confusion_matrix.png"

    plt.savefig(path)

    print(f"\nConfusion matrix saved to {path}")


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():

    print("\nStarting evaluation pipeline...\n")

    model, tokenizer = load_trained_model()

    print("Loading dataset...")
    _, test_dataset = load_training_dataset(tokenizer)

    y_true, y_pred = evaluate_model(model, tokenizer, test_dataset)

    save_confusion_matrix(y_true, y_pred)

    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    main()