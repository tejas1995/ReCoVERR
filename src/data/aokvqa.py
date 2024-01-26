import torch
import os
import json
from collections import defaultdict, Counter
from torch.utils.data import Dataset
from PIL import Image
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

class AOKVQADataset(Dataset):

    def __init__(self, split: str, mode: str ='q2a', vis_processors=None, text_processors=None, type: str='direct'):

        data_dir = '/net/nfs.cirrascale/mosaic/tejass/data/a-okvqa'
        images_dir = '/net/nfs.cirrascale/mosaic/tejass/data/ms-coco/'
        image_filenames = os.listdir(images_dir)
        self.name = 'A-OKVQA'
        self.mode = mode
        self.split = split
        self.type = type
        assert type in ['direct', 'mc']

        if vis_processors is not None:
            self.vis_processor = vis_processors['train'] if split == 'train' else vis_processors['eval']
        if text_processors is not None:
            self.text_processor = text_processors['train'] if split == 'train' else text_processors['eval']

        self.imageid2filename = {}
        for fn in image_filenames:
            image_id = int(fn.split('_')[-1].strip('.jpg'))
            self.imageid2filename[image_id] = os.path.join(images_dir, fn)
        self.imageids = list(set(list(self.imageid2filename.keys())))

        data_file = os.path.join(data_dir, f"aokvqa_v1p0_{split}.json")
        preprocd_data_file = os.path.join(data_dir, 'preprocd_data', f'preprocd_data_{split}.json')
        if os.path.exists(preprocd_data_file) is False:
            data = json.load(open(data_file))
            self.qid2score_dict = {}
            self.data = []
            for i, datum in tqdm(enumerate(data)):
                #if datum['difficult_direct_answer'] is True:
                #    continue
                datum['image_filename'] = self.imageid2filename[datum['image_id']]
                #direct_answers = [postprocess_ok_vqa_generation(lemmatize(a)) for a in datum['direct_answers']]
                direct_answers = datum['direct_answers']
                datum['top_answer'] = max(set(direct_answers), key=direct_answers.count)
                #datum['top_answer'] = datum['choices'][datum['correct_choice_idx']]
                answer_counter = Counter(direct_answers)

                score_dict = defaultdict(float)
                for a, c in answer_counter.items():
                    score_dict[a] = min(1, c/3.0)
                datum['answer_counter'] = answer_counter
                datum['full_score_dict'] = score_dict
                datum['qid'] = i
                self.qid2score_dict[i] = score_dict
                self.data.append(datum)
            json.dump(self.data, open(preprocd_data_file, 'w'), indent=2)
        else:
            self.data = json.load(open(preprocd_data_file))
            if split == 'val':
                self.data = [d for d in self.data if d['difficult_direct_answer'] is False]
            self.qid2score_dict = {d['qid']: d['full_score_dict'] for d in self.data}

        self.qids = list(range(len(self.data)))
        logger.info(f"Loaded A-OKVQA {split} dataset with {len(self.data)} examples!")
        if vis_processors is None or text_processors is None:
            logger.warning("Vision/text processors not set!")

    def set_rationales(self, qid2rationales):
        for i, d in enumerate(self.data):
            rationale = qid2rationales[d['qid']]
            d['rationales'] = [rationale]
        logger.info(f"Set predicted rationales for {len(self.data)} examples!")

    def filter_by_qids(self, qids_list):
        filtered_data = [d for d in self.data if d['qid'] in qids_list]
        self.data = filtered_data
        logger.info(f"Filtered A-OKVQA {self.split} dataset to {len(self.data)} examples!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.data[i]
        image_filename = data['image_filename']
        #image_filename = "/net/nfs.cirrascale/mosaic/tejass/data/black.jpg"
        raw_image = Image.open(image_filename).convert('RGB')
        
        question = data['question']
        if self.type == 'mc':
            question += ' \n Options: ' + ', '.join(data['choices'])
        #answer = data['choices'][ data['correct_choice_idx'] ]
        answer = data['top_answer']
        rationale = data['rationales'][0]
        if self.mode == 'q2a':
            text_input = f"Question: {question} Answer: "
            text_output = answer
        elif self.mode == 'qr2a':
            text_input = f"Question: {question} Rationale: {rationale} Answer: "
            text_output = answer
        elif self.mode == 'q2r':
            text_input = f"Question: {question} Rationale: "
            text_output = rationale
        elif self.mode == 'q2ra':
            text_input = f"Question: {question} Rationale: "
            text_output = f"{rationale} Answer: {answer}"

        choices = data['choices']
        image_id = data['image_id']
        qid = data['qid']
        answer_counter = data['answer_counter']
        correct_choice_idx = data['correct_choice_idx']

        if self.type == 'direct':
            score_dict = defaultdict(float, data['full_score_dict'])
            reference_answers = []
            for a in answer_counter:
                reference_answers += [a]*answer_counter[a]
        elif self.type == 'mc':
            score_dict = defaultdict(float)
            for c in data['choices']:
                score_dict[c] = 0.0
            score_dict[data['choices'][data['correct_choice_idx']]] = 1.0
            reference_answers = [data['choices'][data['correct_choice_idx']]]

        return {
            'text_input': text_input,
            'text_output': text_output,
            'raw_image': raw_image,
            'score_dict': score_dict,
            'image_path': image_filename,
            'image_id': image_id,
            'question': question,
            'top_answer': answer,
            'qid': qid,
            'choices': choices,
            'reference_answers': reference_answers,
            'correct_choice_idx': correct_choice_idx,
        }

    def aokvqa_collate_fn(self, batch):

        images = [b['raw_image'] for b in batch]
        processed_images = torch.stack([self.vis_processor(img) for img in images], dim=0)

        text_inputs = [b['text_input'] for b in batch]
        processed_text_inputs = [self.text_processor(txt) for txt in text_inputs]

        text_outputs = [b['text_output'] for b in batch]
        processed_text_outputs = [self.text_processor(txt) for txt in text_outputs]

        score_dict = [b['score_dict'] for b in batch]
        qids = [b['qid'] for b in batch]
        choices = [b['choices'] for b in batch]
        questions = [b['question'] for b in batch]

        collated_batch = {
            "image": processed_images,
            "text_input": processed_text_inputs,
            "text_output": processed_text_outputs,
            "prompt": text_inputs,
            "target": text_outputs,
            "qids": qids,
            "choices": choices,
            "questions": questions,
        }
        return collated_batch

if __name__ == '__main__':
    dataset = AOKVQADataset('train')
    import pdb; pdb.set_trace()

