import copy
import torch
import os
import json
from collections import defaultdict, Counter
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import random
import logging
from tqdm import tqdm

from utils.vqa_utils import get_score
from utils.okvqa_utils import postprocess_ok_vqa_generation, lemmatize


logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class SherlockDataset(Dataset):

    def __init__(self, split: str, mode: str ='q2a', vis_processors=None, text_processors=None, type: str='direct'):

        data_dir = '/net/nfs.cirrascale/mosaic/tejass/data/sherlock'
        self.vcr_images_dir = '/net/nfs.cirrascale/mosaic/tejass/data/sherlock/vcr1images/'
        self.vg_images_dir = '/net/nfs.cirrascale/mosaic/tejass/data/sherlock/visual_genome/'
        self.name = 'Sherlock'
        self.mode = mode
        self.split = split
        self.type = type
        assert type in ['direct', 'mc']

        if vis_processors is not None:
            self.vis_processor = vis_processors['train'] if split == 'train' else vis_processors['eval']
        if text_processors is not None:
            self.text_processor = text_processors['train'] if split == 'train' else text_processors['eval']

        #instances_file = os.path.join(data_dir, f"{split}_comparison/{split}_comparison_instances.json")
        #answer_file = os.path.join(data_dir, f"{split}_comparison/{split}_comparison_answer_key.json")
        #answer_key = json.load(open(answer_file))
        #inpcandid2anno = {}
        #for x in answer_key['annotations']:
        #    for c in x['candidates']:
        #        inpcandid2anno[(x['Input_iid'], c['source_iid'])] = c
        #testid2inpcandid = {k: (v['Input_iid'], v['candidate']) for k, v in answer_key['test_id_map'].items()}

        preprocd_data_file = os.path.join(data_dir, 'preprocd_data_balanced', f'preprocd_data_{split}.json')
        assert os.path.exists(preprocd_data_file) is True
        self.data = json.load(open(preprocd_data_file))

        logger.info(f"Loaded Sherlock {split} dataset with {len(self.data)} examples!")
        if vis_processors is None or text_processors is None:
            logger.warning("Vision/text processors not set!")

    def url2filepath(self, url):
        if 'VG_' in url:
            return self.vg_images_dir + '/'.join(url.split('/')[-2:])
        else:
            # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
            if 'vcr1images' in self.vcr_images_dir:
                return self.vcr_images_dir + '/'.join(url.split('/')[-2:])
            else:
                return self.vcr_images_dir + '/'.join(url.split('/')[-3:])

    def hide_region(self, image, bboxes, mode):
        image = image.convert('RGBA')
        if mode == 1: # hide mode
            draw = ImageDraw.Draw(image, 'RGBA')
        if mode in [2,5,7,8,9]: #highlight mode
            overlay = Image.new('RGBA', image.size, '#00000000')
            draw = ImageDraw.Draw(overlay, 'RGBA')
        if mode == 3 or mode == 6: #blackout mode or position only mode
            overlay = Image.new('RGBA', image.size, '#7B7575ff')
            draw = ImageDraw.Draw(overlay, 'RGBA')
        for bbox in bboxes:
            x = bbox['left']
            y = bbox['top']
            if mode == 1: # hide mode
                draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])], fill='#7B7575')
            elif mode in [2,5,7,8,9]: # highlight mode
                draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                            outline='#05ff37ff', width=3)
            elif mode == 3: # blackout mode
                draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                            fill='#00000000')
            elif mode == 6: # position only mode
                draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                            fill='#ff05cdff')

        if mode in [2, 3, 5, 6, 7, 8, 9]:
            image = Image.alpha_composite(image, overlay)

        image = image.convert('RGB')
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.data[i]

        #image_filename = "/net/nfs.cirrascale/mosaic/tejass/data/black.jpg"
        image_filename = self.url2filepath(data['image']['url'])
        raw_image = Image.open(image_filename).convert('RGB')
        bboxes = data['region']
        raw_image = self.hide_region(raw_image, bboxes, 2)

        question = f"Is the following statement true about the green highlighted box? {data['inference']} Options: yes, no"
        #answer = data['choices'][ data['correct_choice_idx'] ]
        answer = "yes" if data['label'] == 1 else "no"
        score_dict = defaultdict(float)
        score_dict[answer] = 1.0
        
        qid = data['test_id']
        image_id = image_filename.split('/')[-1].replace('.jpg', '')
        #import pdb; pdb.set_trace()

        return {
            'raw_image': raw_image,
            'qid': qid,
            'image_id': image_id,
            'image_path': image_filename,
            'question': question,
            'top_answer': answer,
            'label': data['label'],
            'bbox': bboxes,
            'score_dict': score_dict, 
            'reference_answers': [answer],
        }


if __name__ == '__main__':
    dataset = SherlockDataset('val')
    import pdb; pdb.set_trace()

