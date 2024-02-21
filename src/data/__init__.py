from .aokvqa import AOKVQADataset
from .vqav2 import VQAv2Dataset
from .okvqa import OKVQADataset
from .sherlock import SherlockDataset

DATASET_REGISTRY = {
    'aokvqa': AOKVQADataset,
    'vqav2': VQAv2Dataset,
    'okvqa': OKVQADataset,
    'sherlock': SherlockDataset
}