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
import spacy
import torch

from data import DATASET_REGISTRY
from models.vlm import VLM_CLASS_MAP
#random.seed(42)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

def determine_if_answer_found(question, predicted_answer):
    '''
    returns True if rollout is able to determine an answer, regardless of correctness
    '''
    prompt = f"The following question was asked to a VQA model, which generated the following answer." \
        f"Question: {question}\n" \
        f"Predicted answer: {predicted_answer}\n\n" \
        f"Was the VQA model able to determine an answer for the above question? Answer yes or no: "
    messages = [{'role': 'user', 'content': prompt}]
    response = openai_caller(messages, model='gpt4', max_new_tokens=3, temperature=0.0)
    response = response.lower().strip()
    assert response in ['yes', 'no']
    return True if response == 'yes' else False

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=DATASET_REGISTRY.keys(), default='aokvqa')
    parser.add_argument("--split", type=str, choices=['train', 'val'], default='val')
    parser.add_argument("--task_type", type=str, choices=['multichoice', 'direct_answer'], default='direct_answer')
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--experiments_dir", type=str, default='/net/nfs.cirrascale/mosaic/tejass/experiments/recoverr/directvqa')
    parser.add_argument("--wandb_config_file", type=str, default="/net/nfs.cirrascale/mosaic/tejass/data/wandb_config.yaml")
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

    # Create agent and environment
    config = yaml.safe_load(open(args.config_file))

    # Load VLM
    vlm_class = config['vlm']['class_name']
    vlm_model_class = VLM_CLASS_MAP[vlm_class]
    vlm_config = yaml.safe_load(open(config['vlm']['model_config_path']))
    vlm_model = vlm_model_class(vlm_config, device)
    vlm_model.set_vqa_inference_params(config['vlm']['vqa_inference_params'])
    vlm_model.set_caption_inference_params(config['vlm']['caption_inference_params'])
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
    #pdb.set_trace()


    eval_score, total_lave_score = 0.0, 0.0
    t = tqdm(range(len(dataset)))
    qids_completed = 0
    qid2rollouts = defaultdict(list)
    if os.path.exists(output_file):
        qid2rollouts = json.load(open(output_file))
        qid2rollouts = defaultdict(list, qid2rollouts)
        eval_score = sum([max([x['score'] for x in v]) for k, v in qid2rollouts.items()])
        total_lave_score = sum([max([x['lave_score'] for x in v]) for k, v in qid2rollouts.items()])
        qids_completed = len(qid2rollouts.keys())
        logger.info(f"Loaded {len(qid2rollouts)} outputs from {output_file}")
        logger.info(f"Eval score: {eval_score/qids_completed*100.0:.2f}%")
    pdb.set_trace()
    for i in t:

        start_time = time.time()
        data = dataset[i]
        qid = data['qid']
        if str(qid) in qid2rollouts.keys():
            wandb_logger.log({})
            continue
        question = data['question']
        image_id = data['image_id']
        image = data['raw_image']
        image_path = data['image_path']
        score_dict = data['score_dict']
        choices = data['choices']

        for rollout_num in range(args.num_rollouts):
            
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
                question=query_details['question']
            )
            if args.dataset == 'okvqa':
                answer = postprocess_ok_vqa_generation(lemmatize(answer))
            chat_history = None
            score = score_dict[answer]

            if args.dataset in ['aokvqa', 'okvqa']:
                lave_reasoning, lave_score = lave_scorer.compute(
                    prediction=answer,
                    references=data['reference_answers'],
                    question=question,
                )
            else:
                lave_score = score
                lave_reasoning = ""

            #answer_found = determine_if_answer_found(question, answer)
            answer_found = True

            qid2rollouts[qid].append(
                    {'qid': qid,
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
                )
            #pdb.set_trace()

        eval_score += max([x['score'] for x in qid2rollouts[qid]])
        total_lave_score += max([x['lave_score'] for x in qid2rollouts[qid]])
        qids_completed += 1
        if qids_completed == args.num_examples:
            break
        json.dump(qid2rollouts, open(output_file, 'w'), indent=2)
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
    print("{} score = {:.2f}%".format(dataset.name, final_score))
    # print(f"{num_missing_answers} answers not contained in choices")
    json.dump(qid2rollouts, open(output_file, 'w'), indent=2)
    print("Total cost = ${:.2f}".format(openai_caller.compute_cost()))
    pdb.set_trace()

if __name__ == '__main__':
    main()