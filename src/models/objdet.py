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

class LVISObjectDetector(nn.Module):
    def __init__(self, config, device):
        super(LVISObjectDetector, self).__init__()

        import detectron2

        # import some common detectron2 utilities
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.data import MetadataCatalog, DatasetCatalog

        self.config = config
        self.device = device
        self.objdet_threshold = config['objdet_threshold']

        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(config['model_name']))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.objdet_threshold  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['model_name'])
        self.predictor = DefaultPredictor(cfg)
        self.class_labels = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

        self.display_name = config['model_display_name']
        num_params = sum(p.numel() for p in self.predictor.model.parameters())

        logger.info(f"Loaded {self.display_name} object detector! ({num_params*10**-9:.2f}B params, not on GPU)")

    def detect(self, image):
        image_array = np.array(image)
        outputs = self.predictor(image_array)
        classes = outputs['instances'].pred_classes.tolist()
        objects = [self.class_labels[i] for i in classes]
        objects = [x.replace('_',' ') for x in set(objects)]
        return objects




OBJDET_CLASS_MAP = {
    'detr': DetrObjectDetector,
    'lvis': LVISObjectDetector
}