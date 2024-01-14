from .aokvqa import AOKVQADataset
#from .vqav2 import VQAv2Dataset
from .okvqa import OKVQADataset

DATASET_REGISTRY = {
    'aokvqa': AOKVQADataset,
    #'vqav2': VQAv2Dataset,
    'okvqa': OKVQADataset,
}