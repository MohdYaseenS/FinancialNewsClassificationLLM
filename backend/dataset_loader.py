from datasets import load_dataset
from configs.config import DATASET_NAME

def load_training_dataset():

    dataset = load_dataset(DATASET_NAME, split='train')

    return dataset