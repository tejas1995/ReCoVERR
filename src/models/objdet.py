import pdb
from typing import List
import pickle as pkl

import numpy as np
import torch
from torch import nn
from transformers import AutoImageProcessor, DetrForObjectDetection

from utils.mem_utils import calculate_model_size_gb

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class DetrObjectDetector(nn.Module):
    def __init__(self, config, device):
        super(DetrObjectDetector, self).__init__()
        self.config = config
        self.device = device
        self.objdet_threshold = config['objdet_threshold']
        self.display_name = config['model_display_name']
        self.image_processor = AutoImageProcessor.from_pretrained(config['model_name'])
        self.model = DetrForObjectDetection.from_pretrained(config['model_name']).to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())

        self.device = device
        self.use_confidence_calibrator = False
        logger.info(f"Loaded {self.display_name} object detector! {calculate_model_size_gb(self.model):.2f}GB in memory")

    def detect(self, image):
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, threshold=self.objdet_threshold, target_sizes=target_sizes)[
            0
        ]
        detected_objects = [self.model.config.id2label[label.item()] for label in results['labels']]
        detected_objects = set(detected_objects)
        return detected_objects

OBJDET_CLASS_MAP = {
    'detr': DetrObjectDetector
}