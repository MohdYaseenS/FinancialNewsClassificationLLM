import torch
import gc

from transformers import TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

from backend.model_loader import load_model
from backend.dataset_loader import load_training_dataset
from configs.config import OUTPUT_DIR, USE_QLORA


def train():

    print("\nStarting training pipeline...\n")

    # ---------------------------------------------------
    # Load model
    # ---------------------------------------------------

    print("Loading model...")

    model, tokenizer = load_model()
    print("Model device:", next(model.parameters()).device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------
    # Load dataset
    # ---------------------------------------------------

    print("Loading dataset...")

    train_dataset, test_dataset = load_training_dataset(tokenizer)

    # ---------------------------------------------------
    # Prepare model for QLoRA
    # ---------------------------------------------------

    if USE_QLORA:
        print("Preparing model for QLoRA...")
        model = prepare_model_for_kbit_training(model)

    # ---------------------------------------------------
    # LoRA configuration
    # ---------------------------------------------------

    print("Applying LoRA...")

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

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=25,
        save_strategy="epoch",
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

    # ---------------------------------------------------
    # Training
    # ---------------------------------------------------

    print("\nStarting fine-tuning...\n")

    trainer.train()

    print("\nFine-tuning complete!")

    # ---------------------------------------------------
    # Save adapter
    # ---------------------------------------------------

    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Adapter and tokenizer saved.")

    # ---------------------------------------------------
    # Clean memory
    # ---------------------------------------------------

    del trainer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    train()