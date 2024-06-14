from .aokvqa import AOKVQADataset
from .vqav2 import VQAv2Dataset

DATASET_REGISTRY = {
    'aokvqa': AOKVQADataset,
    'vqav2': VQAv2Dataset,
}