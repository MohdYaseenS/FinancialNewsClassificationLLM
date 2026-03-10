from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

from backend.model_loader import load_model
from backend.dataset_loader import load_training_dataset
from configs.config import OUTPUT_DIR


def train():

    print("Starting training pipeline...")

    print("Loading model...")
    model, tokenizer = load_model()

    print("Loading dataset...")
    tokenized_dataset = load_training_dataset(tokenizer)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "v_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False
        )
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()