from datasets import load_dataset
from configs.config import DATASET_NAME


def load_training_dataset():

    dataset = load_dataset(DATASET_NAME)

    dataset = dataset["train"]

    def format_prompt(example):

        return {
            "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        }

    dataset = dataset.map(format_prompt)

    return dataset