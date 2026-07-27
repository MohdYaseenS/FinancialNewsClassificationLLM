import torch
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from backend.dataset_loader import load_training_dataset
from backend.model_loader import load_model
from configs.config import OUTPUT_DIR

from tqdm.auto import tqdm
# ---------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------
def extract_prompt(full_text):
    return full_text.split("<|im_start|>assistant")[0]

def extract_true_label(full_text):
    try:
        # isolate assistant block
        assistant_part = full_text.split("<|im_start|>assistant")[1]
        assistant_part = assistant_part.split("<|im_end|>")[0]

        # remove think block completely
        if "<think>" in assistant_part:
            assistant_part = assistant_part.split("</think>")[-1]

        # clean lines
        lines = [l.strip().lower() for l in assistant_part.split("\n") if l.strip()]

        # last valid label
        for line in reversed(lines):
            if line in ["positive", "negative", "neutral"]:
                return line

    except Exception as e:
        print("Label extraction error:", e)

    return "unknown"

def extract_pred_label(text):
    try:
        if "<|im_start|>assistant" in text:
            text = text.split("<|im_start|>assistant")[-1]

        if "<|im_end|>" in text:
            text = text.split("<|im_end|>")[0]

        if "<think>" in text:
            text = text.split("</think>")[-1]

        lines = [l.strip().lower() for l in text.split("\n") if l.strip()]

        for line in reversed(lines):
            if line in ["positive", "negative", "neutral"]:
                return line

    except Exception as e:
        print("Prediction extraction error:", e)

    return "unknown"

def evaluate_model(model, tokenizer, test_dataset):

    print("\nRunning evaluation on test dataset...")

    y_true = []
    y_pred = []

    model.eval()

    for example in tqdm(
        test_dataset,
        desc="Evaluating",
        unit="sample",
        colour="green"
    ):

        full_text = example["text"]
        prompt = extract_prompt(full_text)

        true_label = extract_true_label(full_text)

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
            skip_special_tokens=True
        )

        pred_label = extract_pred_label(prediction)

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

    model, tokenizer = load_model(trainable=False)

    print("Loading dataset...")
    _, test_dataset = load_training_dataset(tokenizer)

    y_true, y_pred = evaluate_model(model, tokenizer, test_dataset)

    save_confusion_matrix(y_true, y_pred)

    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    main()