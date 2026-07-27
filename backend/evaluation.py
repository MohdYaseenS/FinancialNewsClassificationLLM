import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from backend.dataset_loader import load_training_dataset
from backend.model_loader import load_model
from configs.config import OUTPUT_DIR, SYSTEM_PROMPT
from backend.utils import extract_pred_label

from tqdm.auto import tqdm
# ---------------------------------------------------
# Evaluation Logic
# --------------------------------------------------

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

        sentence = example["sentence"]
        true_label = example["label"]
        messages = [
            {
                "role":"user",
                "content":f"{SYSTEM_PROMPT}\nSentence: {sentence}"
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
        pred_label = extract_pred_label(prediction)

        y_pred.append(pred_label)
        y_true.append(true_label)

    return y_true, y_pred


# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------

def save_confusion_matrix(y_true, y_pred):

    labels = ["positive", "negative", "neutral"]

    # ---------------------------------------------------
    # Overall Metrics
    # ---------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\nEvaluation Metrics")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # ---------------------------------------------------
    # Classification Report
    # ---------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    csv_path = f"{OUTPUT_DIR}/classification_report.csv"
    report_df.to_csv(csv_path)

    txt_path = f"{OUTPUT_DIR}/classification_report.txt"

    with open(txt_path, "w") as f:
        f.write(
            classification_report(
                y_true,
                y_pred,
                labels=labels,
                zero_division=0
            )
        )

    print("\nClassification Report\n")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0
        )
    )

    print(f"Classification report saved to {csv_path}")
    print(f"Classification report saved to {txt_path}")

    # ---------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    df_cm = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        df_cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    cm_path = f"{OUTPUT_DIR}/confusion_matrix.png"

    plt.savefig(cm_path)

    print(f"Confusion matrix saved to {cm_path}")


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