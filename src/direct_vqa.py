import os
import time
import json
import pdb
import random
import yaml
from tqdm import tqdm
import argparse
from collections import defaultdict
from typing import List, Union, Dict, Tuple

import numpy as np
import torch

from data import DATASET_REGISTRY
from models.vlm import VLM_CLASS_MAP
#random.seed(42)

from utils.eval_utils import lave_scorer
from utils.openai_utils import openai_caller
from utils.okvqa_utils import postprocess_ok_vqa_generation, lemmatize
from utils.wandb import wandb_logger

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_config_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=DATASET_REGISTRY.keys(), default='aokvqa')
    parser.add_argument("--split", type=str, choices=['train', 'val'], default='val')
    parser.add_argument("--task_type", type=str, choices=['multichoice', 'direct_answer'], default='direct_answer')
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--experiments_dir", type=str, default='/home/tejas/experiments/recoverr_reproduce/directvqa')
    parser.add_argument("--wandb_config_file", type=str, default="/home/tejas/projects/wandb_config.yaml")
    args = parser.parse_args()

    # Create dataset
    dataset_class = DATASET_REGISTRY[args.dataset]
    if args.task_type == 'multichoice':
        dataset = dataset_class(split=args.split, type='mc')
    elif args.task_type == 'direct_answer':
        dataset = dataset_class(split=args.split)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if args.num_examples == -1:
        args.num_examples = len(dataset)

    # Load VLM
    vlm_config = yaml.safe_load(open(args.vlm_config_file))
    vlm_class = vlm_config['model_class']
    vlm_model_class = VLM_CLASS_MAP[vlm_class]
    vlm_model = vlm_model_class(vlm_config, device)
    vlm_model.set_vqa_inference_params(vlm_config['vqa_inference_params'])
    logger.info(f"Finished loading VLM")
    logger.info("-"*100)

    # Create experiment directories
    experiment_dir = os.path.join(args.experiments_dir, f'{args.dataset}_{args.task_type}/{args.split}_outputs')
    experiment_name = vlm_config['model_shorthand'].replace('_', '') + '_direct_vqa'
    experiment_name += '-{}rollouts'.format(args.num_rollouts)
    experiment_name += '-{}examples'.format(args.num_examples)

    wandb_logger.initialize(
        wandb_config_filename=args.wandb_config_file, 
        experiment_name=f'{args.dataset}_{args.split}-{experiment_name}',
        project_name='direct_vqa'
    )
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    output_file = f'{experiment_dir}/{experiment_name}.json'
    logger.info(f"Output file: {output_file}")


    eval_score, total_lave_score = 0.0, 0.0
    t = tqdm(range(len(dataset)))
    qids_completed = 0
    directvqa_rollouts = defaultdict(dict)
    if os.path.exists(output_file):
        directvqa_rollouts = json.load(open(output_file))
        directvqa_rollouts = defaultdict(list, directvqa_rollouts)
        eval_score = sum([v['score'] for k, v in directvqa_rollouts.items()])
        total_lave_score = sum([v['lave_score'] for k, v in directvqa_rollouts.items()])
        qids_completed = len(directvqa_rollouts.keys())
        logger.info(f"Loaded {len(directvqa_rollouts)} outputs from {output_file}")
        logger.info(f"Eval score: {eval_score/qids_completed*100.0:.2f}%")
    for i in t:

        start_time = time.time()
        data = dataset[i]
        qid = data['qid']
        if str(qid) in directvqa_rollouts.keys():
            wandb_logger.log({})
            continue
        question = data['question']
        image_id = data['image_id']
        image = data['raw_image']
        image_path = data['image_path']
        score_dict = data['score_dict']
        #choices = data['choices']

        query_details = {
            'question': question,
            'image': image,
            'qid': qid,
            'image_id': image_id,
            'image_path': image_path,
        }
        # Initialize rollout
        answer, answer_logprobs_dict = vlm_model.ask(
            raw_image=query_details['image'], 
            question=query_details['question'],
            query=query_details,
        )
        if args.dataset == 'okvqa':
            answer = postprocess_ok_vqa_generation(lemmatize(answer))
        score = score_dict[answer]

        if args.dataset in ['aokvqa', 'okvqa', 'vqav2']:
            lave_reasoning, lave_score = lave_scorer.compute(
                prediction=answer,
                references=data['reference_answers'],
                question=question,
            )
        else:
            lave_score = score
            lave_reasoning = ""

        directvqa_rollouts[qid] = {
                'qid': qid,
                'image_id': data['image_id'],
                'image_path': data['image_path'], 
                'vqa_question': question, 
                'annotated_answers': data['reference_answers'],
                'answer': answer, 
                'answer_logprobs_dict': answer_logprobs_dict,
                'score': score, 
                'lave_score': lave_score,
                'lave_reasoning': lave_reasoning,
                'risk': 1 - lave_score,
            }
        #pdb.set_trace()

        eval_score += directvqa_rollouts[qid]['score']
        total_lave_score += directvqa_rollouts[qid]['lave_score']
        qids_completed += 1
        if qids_completed == args.num_examples:
            break
        json.dump(directvqa_rollouts, open(output_file, 'w'), indent=2)
        wandb_logger.log({
            'qid': qid,
            'eval_score': 100.0*eval_score/qids_completed,
            'num_samples': qids_completed,
            'total_cost': openai_caller.compute_cost(),
            'time_per_sample': time.time() - start_time,
            'lave_score':100.0*total_lave_score/qids_completed,
        })
        t.set_description(f"Evaluating {dataset.name} (VQAAcc score = {eval_score/qids_completed:.2%}, LAVE score = {total_lave_score/qids_completed:.2%}, cost=${openai_caller.compute_cost():.2f})")

    final_score = 100.0*eval_score/qids_completed
    logger.info("{} score = {:.2f}%".format(dataset.name, final_score))
    json.dump(directvqa_rollouts, open(output_file, 'w'), indent=2)
    logger.info(f"Saved {len(directvqa_rollouts)} outputs to {output_file}")
    logger.info("Total cost = ${:.2f}".format(openai_caller.compute_cost()))
    pdb.set_trace()

if __name__ == '__main__':
    main()