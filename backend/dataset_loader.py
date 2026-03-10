from datasets import load_dataset
from configs.config import DATASET_NAME


def load_training_dataset(tokenizer):

    dataset = load_dataset(DATASET_NAME)["train"]

    def format_prompt(example):

        system_prompt = (
            "Classify the sentiment of the following sentence from News "
            "as positive, negative, or neutral."
        )

        user_prompt = f"Sentence: {example['text']}"

        assistant_response = example["sentiment"]

        messages = [
            {
                "role": "user",
                "content": f"{system_prompt}\n{user_prompt}"
            },
            {
                "role": "assistant",
                "content": assistant_response
            }
        ]

        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        return {"text": formatted_text}

    dataset = dataset.map(format_prompt)

    return dataset