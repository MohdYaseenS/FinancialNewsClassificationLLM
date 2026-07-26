import torch
import gc
import os

from transformers import TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

from backend.model_loader import load_model
from backend.dataset_loader import load_training_dataset
from backend.evaluation import evaluate_model

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from configs.config import OUTPUT_DIR, USE_QLORA


def train():

    print("\nStarting training pipeline...\n")

    # ---------------------------------------------------
    # Load model
    # ---------------------------------------------------

    print("Loading model...")
    model, tokenizer = load_model(trainable=True)

    # ---------------------------------------------------
    # Prepare model for QLoRA
    # ---------------------------------------------------

    if USE_QLORA:
        print("Preparing model for QLoRA...")
        model = prepare_model_for_kbit_training(model)

    adapter_exists = os.path.exists(
    os.path.join(OUTPUT_DIR, "adapter_config.json")
    )
    # ---------------------------------------------------
    # Load dataset
    # ---------------------------------------------------

    print("Loading dataset...")
    train_dataset, test_dataset = load_training_dataset(tokenizer)

    # ---------------------------------------------------
    # LoRA config (only if fresh training)
    # ---------------------------------------------------

    lora_config = None

    if not adapter_exists:
        print("Applying fresh LoRA configuration...")

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

    # ---------------------------------------------------
    # Training arguments
    # ---------------------------------------------------

    num_epochs = 10  # 🔥 change as needed

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=1,  # IMPORTANT: we control epochs manually
        logging_steps=25,
        save_strategy="no",  # we save manually
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        warmup_steps=0.03,
        lr_scheduler_type="constant",
        report_to="none"
    )

    # ---------------------------------------------------
    # Trainer
    # ---------------------------------------------------

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=lora_config,
        processing_class=tokenizer
    )

    print("\nTrainer initialized.")

    try:
        model.print_trainable_parameters()
    except:
        pass

    # ---------------------------------------------------
    # Training Loop (manual epochs)
    # ---------------------------------------------------

    print("\nStarting fine-tuning...\n")

    eval_history = []

    for epoch in range(num_epochs):

        print(f"\n========== Epoch {epoch+1} / {num_epochs} ==========")

        trainer.train()

        # ---------------------------
        # Get training loss
        # ---------------------------

        loss = None
        for log in reversed(trainer.state.log_history):
            if "loss" in log:
                loss = log["loss"]
                break

        print(f"Training Loss: {loss}")

        # ---------------------------
        # Evaluation
        # ---------------------------

        print("Running evaluation...")

        y_true, y_pred = evaluate_model(model, tokenizer, test_dataset)

        acc = accuracy_score(y_true, y_pred)

        print(f"Validation Accuracy: {acc}")

        eval_history.append({
            "epoch": epoch + 1,
            "accuracy": acc,
            "loss": loss
        })

    print("\nFine-tuning complete!")

    # ---------------------------------------------------
    # Save adapter
    # ---------------------------------------------------

    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Adapter and tokenizer saved.")

    # ---------------------------------------------------
    # Plot metrics
    # ---------------------------------------------------

    print("\nPlotting metrics...")

    epochs = [x["epoch"] for x in eval_history]
    accs = [x["accuracy"] for x in eval_history]
    losses = [x["loss"] for x in eval_history]

    plt.figure()

    plt.plot(epochs, accs, marker='o', label="Accuracy")
    plt.plot(epochs, losses, marker='o', label="Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss & Validation Accuracy")
    plt.legend()

    path = f"{OUTPUT_DIR}/training_curve.png"
    plt.savefig(path)

    print(f"Plot saved to {path}")

    # ---------------------------------------------------
    # Cleanup
    # ---------------------------------------------------

    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    print("\nTraining pipeline complete.\n")


if __name__ == "__main__":
    train()