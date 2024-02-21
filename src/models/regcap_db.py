from typing import Dict, List
from PIL import Image
import yaml
import re
import pdb
import jsonlines

import torch
from torch import nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

def filter_bad_regioncaps(all_regioncaps: List[str]) -> List[str]:
    filtered_caps = [c for c in all_regioncaps if re.search('[a-zA-Z]+', c) is not None]
    #if len(filtered_caps) != len(all_regioncaps):
    #    pdb.set_trace()
    return filtered_caps

class QwenVLRegionCaptionDB(nn.Module):
    def __init__(self, config: Dict, device):

        super(QwenVLRegionCaptionDB, self).__init__()
        self.region_captions_file = config['region_captions_file']
        self.db_display_name = config['imcapdb_display_name']
        region_captions_data = jsonlines.open(self.region_captions_file)
        self.imageid2regioncaps = {x['image_id']: filter_bad_regioncaps([b['caption'] for b in x['boxes']]) for x in region_captions_data}
        logger.info(f"Loaded {self.db_display_name} database")

    def get_captions_by_imageid(self, image_id: str) -> str:
        try:
            if 'okvqa' in self.region_captions_file:
                image_id = int(image_id)
            region_captions = self.imageid2regioncaps[image_id]
            region_captions = [c.lower().replace('.', '') for c in region_captions]
            region_captions = [c for c in region_captions if len(c) > 0]
            region_captions = region_captions#[:5]
        except:
            region_captions = []
        return region_captions

REGCAPDB_CLASS_MAP = {
    'qwenvl': QwenVLRegionCaptionDB
}