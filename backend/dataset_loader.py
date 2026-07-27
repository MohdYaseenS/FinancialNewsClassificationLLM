from datasets import load_dataset
from configs.config import DATASET_NAME
from configs.config import SYSTEM_PROMPT


def load_training_dataset(tokenizer):

    dataset = load_dataset(DATASET_NAME)["train"]

    split_dataset = dataset.train_test_split(
        test_size=0.8,
        shuffle=True,
        seed=42
    )

    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    def format_prompt(example):

        system_prompt = SYSTEM_PROMPT
        user_prompt = f"Sentence: {example['text']}"

        label = example["sentiment"].strip().lower()

        messages = [
            {
                "role": "user",
                "content": f"{system_prompt}\n{user_prompt}"
            },
            {
                "role": "assistant",
                "content": label
            }
        ]

        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        return {
            "text": formatted_text,
            "sentence": example["text"],
            "label": label
        }

    train_dataset = train_dataset.map(
        format_prompt,
        remove_columns=list(train_dataset.features)
    )

    test_dataset = test_dataset.map(
        format_prompt,
        remove_columns=list(test_dataset.features)
    )

    return train_dataset, test_dataset