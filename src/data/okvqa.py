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

class OKVQADataset(Dataset):

    def __init__(self, split, mode='q2a', vis_processors=None, text_processors=None, type='direct'):

        data_dir = '/net/nfs.cirrascale/mosaic/tejass/data/okvqa'
        images_dir = '/net/nfs.cirrascale/mosaic/tejass/data/ms-coco/'
        image_filenames = os.listdir(images_dir)
        self.name = 'OKVQA'
        self.mode = mode
        self.split = split
        self.type = type
        assert type in ['direct']

        if vis_processors is not None:
            self.vis_processor = vis_processors['train'] if split == 'train' else vis_processors['eval']
        if text_processors is not None:
            self.text_processor = text_processors['train'] if split == 'train' else text_processors['eval']

        self.imageid2filename = {}
        for fn in image_filenames:
            image_id = int(fn.split('_')[-1].strip('.jpg'))
            self.imageid2filename[image_id] = os.path.join(images_dir, fn)
        self.imageids = list(set(list(self.imageid2filename.keys())))

        self.question_data = json.load(open(os.path.join(data_dir, f"OpenEnded_mscoco_{split}2014_questions.json")))['questions']
        self.anno_data = json.load(open(os.path.join(data_dir, f"mscoco_{split}2014_annotations.json")))['annotations']
        
        qid2qdata = {x['question_id']: x for x in self.question_data}
        preprocd_data_file = os.path.join(data_dir, 'preprocd_data', f'preprocd_data_{split}.json')
        if os.path.exists(preprocd_data_file) is False:
            self.data = []
            self.qid2score_dict = {}
            for datum in tqdm(self.anno_data):
                image_id = datum['image_id']
                image_filename = self.imageid2filename[image_id]
                qid = datum['question_id']
                question = qid2qdata[qid]['question']

                #answers = [postprocess_ok_vqa_generation(lemmatize(a['answer'])) for a in d['answers']]
                direct_answers = [a['answer'] for a in datum['answers']]
                #mport pdb; pdb.set_trace()
                top_answer = max(set(direct_answers), key=direct_answers.count)
                #datum['top_answer'] = datum['choices'][datum['correct_choice_idx']]
                answer_counter = Counter(direct_answers)
                score_dict = defaultdict(float)
                for a, c in answer_counter.items():
                    score_dict[a] = min(1, c/3.0)
                
                d = {
                    'qid': int(qid),
                    'image_id': image_id,
                    'image_filename': image_filename,
                    'question': question,
                    'full_score_dict': score_dict,
                    'top_answer': top_answer,
                    'answer_counter': answer_counter,
                }
                self.data.append(d)
                self.qid2score_dict[int(qid)] = score_dict
            json.dump(self.data, open(preprocd_data_file, 'w'), indent=2)
        else:
            self.data = json.load(open(preprocd_data_file))

        self.qids = list(range(len(self.data)))
        logger.info(f"Loaded OKVQA {split} dataset with {len(self.data)} examples!")
        if vis_processors is None or text_processors is None:
            logger.warning("Vision/text processors not set!")

    def filter_by_qids(self, qids_list):
        filtered_data = [d for d in self.data if d['qid'] in qids_list]
        self.data = filtered_data
        logger.info(f"Filtered OKVQA {self.split} dataset to {len(self.data)} examples!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.data[i]
        image_filename = data['image_filename']
        raw_image = Image.open(image_filename).convert('RGB')
        
        question = data['question']
        answer = data['top_answer']
        text_input = f"Question: {question} Answer: "
        text_output = answer

        score_dict = defaultdict(float, data['full_score_dict'])
        image_id = data['image_id']
        qid = data['qid']
        answer_counter = data['answer_counter']
        reference_answers = []
        for a in answer_counter:
            reference_answers += [a]*answer_counter[a]

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
            'choices': list(score_dict.keys()),
            'reference_answers': reference_answers,
            'correct_choice_idx': 0,
        }

    def okvqa_collate_fn(self, batch):

        images = [b['raw_image'] for b in batch]
        processed_images = torch.stack([self.vis_processor(img) for img in images], dim=0)

        text_inputs = [b['text_input'] for b in batch]
        processed_text_inputs = [self.text_processor(txt) for txt in text_inputs]

        text_outputs = [b['text_output'] for b in batch]
        processed_text_outputs = [self.text_processor(txt) for txt in text_outputs]

        score_dict = [b['score_dict'] for b in batch]
        qids = [b['qid'] for b in batch]

        collated_batch = {
            "image": processed_images,
            "text_input": processed_text_inputs,
            "text_output": processed_text_outputs,
            "prompt": text_inputs,
            "target": text_outputs,
            "qids": qids,
        }
        return collated_batch

if __name__ == '__main__':
    dataset = OKVQADataset('train')
    import pdb; pdb.set_trace()

