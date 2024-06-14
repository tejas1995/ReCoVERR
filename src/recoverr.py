import os
import copy
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

from models.llm import LLM_CLASS_MAP
from models.vlm import VLM_CLASS_MAP
from models.qgen import QGEN_CLASS_MAP
from models.objdet import OBJDET_CLASS_MAP

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

def run_recoverr_verifyingevidences(
    query_details, 
    models_dict, 
    recoverr_config, 
    do_print=False
) -> Dict:

    do_recoverr = recoverr_config['do_recoverr']
    max_evidence_collection_turns = recoverr_config['max_evidence_collection_turns']
    questions_generated_per_turn = recoverr_config['questions_generated_per_turn']
    desired_risk = recoverr_config['desired_risk']
    vqaconfthresh_at_risk = recoverr_config['vqaconfthresh_at_risk']
    min_entailment_conf = recoverr_config['min_entailment_conf']
    defeasibility_delta = recoverr_config['defeasibility_delta']
    visual_tools = recoverr_config['visual_tools']
    evidence_conf_threshold = 1-desired_risk #+ 0.05
    vlm_conf_type = recoverr_config['vlm_conf_type'] if 'vlm_conf_type' in recoverr_config else 'self_prompting_conf'

    vlm_model = models_dict['vlm']
    objdet_model = models_dict['objdet']
    qgen_model = models_dict['qgen']
    llm_model = models_dict['llm']

    #region DIRECTVQA    
    directvqa_predicted_answer, directvqa_answer_logprobsdict = vlm_model.ask(
            raw_image=query_details['image'], 
            question=query_details['question']
        )
    directvqa_conf = directvqa_answer_logprobsdict[vlm_conf_type]
    directvqa_hypothesis = llm_model.rephrase_qa_to_statement(
        question=query_details['question'],
        answer=directvqa_predicted_answer
    )
    directvqa_evidence = {
        'question': query_details['question'], 
        'answer': directvqa_predicted_answer, 
        'vlm_conf': directvqa_conf, 
        'statement': directvqa_hypothesis, 
        'is_reliable': True if directvqa_conf >= 1-desired_risk else False, 
    }
    if do_print:
        print(f"DirectVQA prediction: {directvqa_predicted_answer} with confidence {directvqa_conf:.4f}")
        print(f"DirectVQA hypothesis: {directvqa_hypothesis}")
    if directvqa_conf >= vqaconfthresh_at_risk:
        if do_print:
            print(f"DirectVQA prediction is reliable, returning it")
        return {
            'hypothesis': directvqa_hypothesis,
            'prediction': directvqa_predicted_answer,
            'prediction_type': 'directvqa',
            'prediction_entailment_conf': 1.0,
            'visual_conf': directvqa_conf, 
            'overall_conf': directvqa_conf, 
            'all_evidences': [], 
            'reliable_evidences': [], 
            'reliable_and_relevant_evidences': [],            
        }
    #endregion

    initial_evidences = []
    #region CREATE CAPTION EVIDENCE
    caption, caption_logprobs_dict = vlm_model.caption(query_details['image'])
    #caption_conf = caption_logprobs_dict[vlm_conf_type]
    caption_conf = 1.0
    caption_entailment_conf = llm_model.get_entailment_confidence(
        premise=caption,
        hypothesis=directvqa_hypothesis
    )
    if do_print:
        print("-"*100)
        print(f"Image caption: {caption} (confidence={caption_conf:.4f})")
        print(f"P({directvqa_hypothesis} | {caption}) = {caption_entailment_conf:.4f}")
    caption_evidence = {
        'question': 'Describe the image',
        'answer': caption,
        'vlm_conf': caption_conf,
        'statement': caption, 
        'prediction_entailment_conf': caption_entailment_conf,
        'counterfactual_prediction_entailment_conf': 0,
        'relevance': caption_entailment_conf, 
        'is_reliable': True, #if caption_conf >= 1-desired_risk else False,
        'is_relevant': True,
    }
    initial_evidences.append(caption_evidence)
    #endregion

    all_evidences = copy.deepcopy([caption_evidence])
    reliable_evidences = copy.deepcopy([caption_evidence])
    reliable_and_relevant_evidences = copy.deepcopy([caption_evidence])

    # region EXTRACT OBJECTS FROM IMAGE
    if 'objdet' in visual_tools:
        image_objects = objdet_model.detect(query_details['image'])

        for object in image_objects:
            object_presence_question = f"Does the image contain a {object}?"
            object_presence_confidence = vlm_model.get_answer_confidence(
                raw_image=query_details['image'], 
                questions=[object_presence_question],
                answers=['yes']
            )[0][0]
            object_statement = f"The image contains a {object}."
            object_evidence = {
                    'question': object_presence_question,
                    'answer': 'yes',
                    'vlm_conf': object_presence_confidence,
                    'statement': object_statement,
                    'prediction_entailment_conf': 0.0,
                    'counterfactual_prediction_entailment_conf': 0.0,
                    'relevance': 0.0,
                    'is_reliable': True if object_presence_confidence >= evidence_conf_threshold else False,
                    'is_relevant': False,
                }
            if object_presence_confidence >= evidence_conf_threshold:
                object_presence_entailment_conf = llm_model.get_entailment_confidence(
                    premise=object_statement,
                    hypothesis=directvqa_hypothesis
                )
                object_absence_premise = f"The image does not contain a {object}."
                object_absence_entailment_conf = llm_model.get_entailment_confidence(
                    premise=object_absence_premise,
                    hypothesis=directvqa_hypothesis
                )
                object_relevance = np.abs(object_presence_entailment_conf - object_absence_entailment_conf)
                object_evidence['prediction_entailment_conf'] = object_presence_entailment_conf
                object_evidence['counterfactual_prediction_entailment_conf'] = object_absence_entailment_conf
                object_evidence['relevance'] = object_relevance
                object_evidence['is_relevant'] = True if object_relevance >= defeasibility_delta else False
                if do_print:
                    print("")
                    print(object_statement)
                    print(f"P({directvqa_hypothesis} | {object_statement}) = {object_presence_entailment_conf:.4f}")
                    print(f"P({directvqa_hypothesis} | {object_absence_premise}) = {object_absence_entailment_conf:.4f}")
    
                
                reliable_evidences.append(object_evidence)
                if object_relevance >= defeasibility_delta:
                    reliable_and_relevant_evidences.append(object_evidence)
            all_evidences.append(object_evidence)
    #endregion
    if do_print:
        print("-"*100)

    # region CHECK CONF BASED ON VERBALIZATIONS
    # Answer only based on the image verbalizations collected so far, no further evidence collection
    if len(reliable_and_relevant_evidences) > 0:
        aggregated_evidences_premise = ' '.join([x['statement'] for x in reliable_and_relevant_evidences])
        aggregated_evidences_nliconf = \
            llm_model.get_entailment_confidence(
            #vlm_model.get_entailment_confidence( 
            #image=query_details['image'],
            premise=aggregated_evidences_premise,
            hypothesis=directvqa_hypothesis
        )#[0][0]
        if do_print:
            print(f"P({directvqa_hypothesis} | {aggregated_evidences_premise}) = {aggregated_evidences_nliconf:.4f}")

        #evidence_conf = min([x['vlm_conf'] for x in reliable_and_relevant_evidences])
        evidence_conf = min([x['vlm_conf'] for x in reliable_and_relevant_evidences])
        expected_conf = evidence_conf*aggregated_evidences_nliconf
        if do_print:
            print(f"Min evidence conf = {evidence_conf:.4f}")
            print(f"Expected conf = {expected_conf:.4f}")
        #if expected_conf >= (1-desired_risk):
        if aggregated_evidences_nliconf >= min_entailment_conf:
            return {
                'hypothesis': directvqa_hypothesis,
                'prediction': directvqa_predicted_answer,
                'prediction_type': 'recoverr_verbalizations',
                'prediction_entailment_conf': aggregated_evidences_nliconf,
                'visual_conf': evidence_conf,
                'overall_conf': aggregated_evidences_nliconf, 
                'all_evidences': initial_evidences, 
                'reliable_evidences': initial_evidences, 
                'reliable_and_relevant_evidences': initial_evidences,
            }
    #endregion

    if not do_recoverr:
            return {
                'hypothesis': directvqa_hypothesis,
                'prediction': "unknown",
                'prediction_type': 'abstained',
                'prediction_entailment_conf': 1.0,
                'visual_conf': 1.0,
                'overall_conf': 1.0, 
                'all_evidences': all_evidences, 
                'reliable_evidences': reliable_evidences, 
                'reliable_and_relevant_evidences': reliable_and_relevant_evidences,
            }

    for j in range(max_evidence_collection_turns):
        #region FIND_RELIABLE_EVIDENCES
        # Gather different evidences about the image, retain only the reliable ones
        if do_print:
            print(f"{'-'*100}\nEvidence collection turn {j}")
        candidate_questions = qgen_model.generate_supportingevidence_questions(
            target_question=query_details['question'], 
            evidences=reliable_evidences, 
            num_questions=questions_generated_per_turn, 
            possible_answer=directvqa_predicted_answer
        )
        try:
            candidate_vqa_answers, vqa_answer_logprobsdicts = vlm_model.ask_multiplequestions(
                raw_image=query_details['image'],
                questions=candidate_questions
            )
        except Exception as e:
            #print(e)
            #pdb.set_trace()
            continue
        candidate_vqa_confidences = [x[vlm_conf_type] for x in vqa_answer_logprobsdicts]
        candidate_vqa_statements = llm_model.rephrase_batched_qas_to_statements(
            questions=candidate_questions,
            answers=candidate_vqa_answers
        )
        #endregion

        # Identify evidences that are reliable AND relevant
        for i in range(len(candidate_questions)):

            # Ignore new evidences that had previously been (reliably) obtained
            if candidate_questions[i]  in [x['question'] for x in reliable_evidences]:
                continue

            if do_print:
                print(f"\nQuestion: {candidate_questions[i]} Answer: {candidate_vqa_answers[i]} (conf={candidate_vqa_confidences[i]:.4f})")
                print(f"Statement: {candidate_vqa_statements[i]}")
            evidence = {
                    'question': candidate_questions[i],
                    'answer': candidate_vqa_answers[i],
                    'vlm_conf': candidate_vqa_confidences[i],
                    'statement': candidate_vqa_statements[i],
                    'is_reliable': True if candidate_vqa_confidences[i] >= evidence_conf_threshold else False, 
                    'is_relevant': False,
                }

            # Only retain a VQA evidence if it is "reliable"
            if candidate_vqa_confidences[i] >= evidence_conf_threshold:
                # Check if evidence is also relevant
                #pdb.set_trace()

                # Estimate confidence of hypothesis based on evidence
                true_evidence_premise = evidence['statement']
                true_evidence_nliconf = llm_model.get_entailment_confidence(
                    premise=true_evidence_premise,
                    hypothesis=directvqa_hypothesis
                )

                # Estimate confidence of hypothesis IF the evidence is false
                false_evidence = copy.deepcopy(evidence)
                false_evidence['answer'] = f"not {false_evidence['answer']}"
                false_evidence_premise = llm_model.rephrase_qa_to_statement(
                    question=false_evidence['question'],
                    answer=false_evidence['answer']
                )
                false_evidence_nliconf = llm_model.get_entailment_confidence(
                    premise=false_evidence_premise,
                    hypothesis=directvqa_hypothesis
                )

                evidence['prediction_entailment_conf'] = true_evidence_nliconf
                evidence['counterfactual_prediction_entailment_conf'] = false_evidence_nliconf
                evidence['relevance'] = np.abs(true_evidence_nliconf - false_evidence_nliconf)
                if do_print:
                    print(f"P({directvqa_hypothesis} | {true_evidence_premise}) = {true_evidence_nliconf:.4f}")
                    print(f"P({directvqa_hypothesis} | {false_evidence_premise}) = {false_evidence_nliconf:.4f}")    

                #if agent_guess_conf > min_entailment_conf and \
                if evidence['relevance'] >= defeasibility_delta:
                    reliable_and_relevant_evidences.append(evidence)
                    evidence['is_relevant'] = True
                    if do_print:
                        print("RELEVANT")
                else:
                    if do_print:
                        print("NOT RELEVANT")
                    pass
                #print()

                reliable_evidences.append(evidence)
            all_evidences.append(evidence)

        if do_print:
            print(f"After evidence collection turn {j}:")
            print(f"Reliable evidences: {len(reliable_evidences)}")
            print(f"Reliable and relevant evidences: {len(reliable_and_relevant_evidences)}")

        # Try to get confidence in original guess based on all reliable and relevant evidences so far
        if len(reliable_and_relevant_evidences) > 0:
            aggregated_evidences_premise = ' '.join([x['statement'] for x in reliable_and_relevant_evidences])
            aggregated_evidences_nliconf = \
                llm_model.get_entailment_confidence(
                #vlm_model.get_entailment_confidence(
                #image=query_details['image'],
                premise=aggregated_evidences_premise,
                hypothesis=directvqa_hypothesis
            )#[0][0]
            if do_print:
                print(f"P({directvqa_hypothesis} | {aggregated_evidences_premise}) = {aggregated_evidences_nliconf:.4f}")

            #evidence_conf = min([x['vlm_conf'] for x in reliable_and_relevant_evidences])
            evidence_conf = min([x['vlm_conf'] for x in reliable_and_relevant_evidences])
            expected_conf = evidence_conf*aggregated_evidences_nliconf
            if do_print:
                print(f"Min evidence conf = {evidence_conf:.4f}")
                print(f"Expected conf = {expected_conf:.4f}")
            #if expected_conf >= (1-desired_risk):
            if aggregated_evidences_nliconf >= min_entailment_conf:
                return {
                    'hypothesis': directvqa_hypothesis,
                    'prediction': directvqa_predicted_answer,
                    'prediction_type': 'recoverr_qa',
                    'prediction_entailment_conf': aggregated_evidences_nliconf,
                    'visual_conf': evidence_conf,
                    'overall_conf': expected_conf, 
                    'all_evidences': all_evidences, 
                    'reliable_evidences': reliable_evidences, 
                    'reliable_and_relevant_evidences': reliable_and_relevant_evidences,
                }


    return {
                'hypothesis': directvqa_hypothesis,
                'prediction': "unknown",
                'prediction_type': 'abstained',
                'prediction_entailment_conf': 1.0,
                'visual_conf': 1.0,
                'overall_conf': 1.0, 
                'all_evidences': all_evidences, 
                'reliable_evidences': reliable_evidences, 
                'reliable_and_relevant_evidences': reliable_and_relevant_evidences,
            }

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=DATASET_REGISTRY.keys(), default='aokvqa')
    parser.add_argument("--split", type=str, choices=['train', 'val'], default='val')
    parser.add_argument("--task_type", type=str, choices=['multichoice', 'direct_answer'], default='direct_answer')
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--experiments_dir", type=str, default='/home/tejas/experiments/recoverr_reproduce/recoverr/')
    parser.add_argument("--wandb_config_file", type=str, default="/home/tejas/projects/wandb_config.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    # Load question generation model
    qgen_class = config['qgen']['class_name']
    qgen_model_class = QGEN_CLASS_MAP[qgen_class]
    qgen_config = yaml.safe_load(open(config['qgen']['model_config_path']))
    qgen_model = qgen_model_class(qgen_config, device)
    logger.info(f"Finished loading QGen")
    logger.info("-"*100)

    # Load VLM
    vlm_class = config['vlm']['class_name']
    vlm_model_class = VLM_CLASS_MAP[vlm_class]
    vlm_config = yaml.safe_load(open(config['vlm']['model_config_path']))
    vlm_model = vlm_model_class(vlm_config, device)
    vlm_model.set_vqa_inference_params(config['vlm']['vqa_inference_params'])
    vlm_model.set_caption_inference_params(config['vlm']['caption_inference_params'])
    if config['vlm']['use_confidence_calibrator'] is True:
        vlm_model.set_confidence_calibrator(config['vlm']['calibrator_model_path'])
    logger.info(f"Finished loading VLM")
    logger.info("-"*100)

    # Load Object detector
    objdet_class = config['objdet']['class_name']
    objdet_model_class = OBJDET_CLASS_MAP[objdet_class]
    objdet_config = yaml.safe_load(open(config['objdet']['model_config_path']))
    objdet_model = objdet_model_class(objdet_config, device)
    logger.info(f"Finished loading ObjectDetector")
    logger.info("-"*100)

    # Load LLM
    llm_class = config['llm']['class_name']
    llm_model_class = LLM_CLASS_MAP[llm_class]
    llm_config = yaml.safe_load(open(config['llm']['model_config_path']))
    llm_model = llm_model_class(llm_config, device)
    logger.info(f"Finished loading LLM")
    logger.info("-"*100)

    models_dict = {
        'vlm': vlm_model,
        'objdet': objdet_model,
        'qgen': qgen_model,
        'llm': llm_model,
    }


    # Create experiment directories
    recoverr_config = config['recoverr']
    experiment_name = config['exp_name']
    experiment_name += '-{}maxcollectionturns-{}questionsperturn'.format(
        recoverr_config['max_evidence_collection_turns'],
        recoverr_config['questions_generated_per_turn'],
    )
    experiment_name += '-{}desiredrisk-{}gamma-{}defeasibilitydelta-{}minentailmentconf'.format(
        recoverr_config['desired_risk'],
        recoverr_config['vqaconfthresh_at_risk'],
        recoverr_config['defeasibility_delta'],
        recoverr_config['min_entailment_conf']
    )
    folder_name = config['folder_name']
    if args.num_rollouts != 1:
        experiment_name += '-{}rollouts'.format(args.num_rollouts)
    if args.num_examples != len(dataset):
        experiment_name += '-{}examples'.format(args.num_examples)
    experiment_name += '-seed{}'.format(args.seed)
    experiment_dir = os.path.join(args.experiments_dir, f'{args.dataset}_{args.task_type}/{args.split}_outputs/recoverr_verifyingevidences_rollouts/{folder_name}')

    wandb_logger.initialize(
        wandb_config_filename=args.wandb_config_file, 
        experiment_name=f'{args.dataset}_{args.split}-{folder_name}-{experiment_name}',
        #project_name='beamsearch_vqa'
        project_name='recoverr_013024'
    )
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    output_file = f'{experiment_dir}/{experiment_name}.json'
    logger.info(f"Output file: {output_file}")
    #pdb.set_trace()

    eval_score, total_lave_score = 0.0, 0.0
    #t = tqdm(range(len(dataset)))
    t = tqdm(range(len(dataset)))
    qids_completed = 0
    recoverr_rollouts = defaultdict(dict)
    questions_answered = 0
    total_risk = 0
    if os.path.exists(output_file):
        recoverr_rollouts = json.load(open(output_file))
        recoverr_rollouts = defaultdict(dict, recoverr_rollouts)
        eval_score = sum([v['score'] for k, v in recoverr_rollouts.items()])
        total_lave_score = sum([v['lave_score'] for k, v in recoverr_rollouts.items()])
        questions_answered += sum(max([x['question_answered'] for x in v]) for k, v in recoverr_rollouts.items())
        total_risk += sum(min([1-x['lave_score'] for x in v if x['question_answered'] == 1], default=0) for k, v in recoverr_rollouts.items())
        qids_completed = len(recoverr_rollouts.keys())
        logger.info(f"Loaded {len(recoverr_rollouts)} outputs from {output_file}")
        logger.info(f"Eval score: {eval_score/qids_completed*100.0:.2f}%")
    #pdb.set_trace()
    for i in t:

        start_time = time.time()
        data = dataset[i]
        qid = data['qid']
        if str(qid) in recoverr_rollouts.keys():
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
        result_dict = run_recoverr_verifyingevidences(
            query_details=query_details,
            models_dict=models_dict,
            recoverr_config=recoverr_config,
            do_print=False
        )
        predicted_answer = result_dict['prediction']
        if args.dataset == 'okvqa':
            predicted_answer = postprocess_ok_vqa_generation(lemmatize(predicted_answer))
        prediction_type = result_dict['prediction_type']
        prediction_entailment_conf = result_dict['prediction_entailment_conf']
        overall_conf = result_dict['overall_conf']
        gathered_evidences = result_dict['all_evidences']
        selected_evidences = result_dict['reliable_and_relevant_evidences']

        score = score_dict[predicted_answer]

        lave_reasoning, lave_score = lave_scorer.compute(
            prediction=predicted_answer,
            references=data['reference_answers'],
            question=question,
        )

        recoverr_rollouts[qid] = {
            'qid': qid,
            'image_id': data['image_id'],
            'image_path': data['image_path'], 
            'vqa_question': question, 
            'annotated_answers': data['reference_answers'],
            'answer': predicted_answer, 
            'score': score, 
            'lave_score': lave_score,
            'lave_reasoning': lave_reasoning,
            'gathered_evidences': gathered_evidences,
            'selected_evidences': selected_evidences,
            'hypothesis': result_dict['hypothesis'],
            'prediction_type': prediction_type, 
            'prediction_entailment_conf': prediction_entailment_conf,
            'overall_conf': overall_conf,
            'question_answered': 0 if prediction_type == 'abstained' else 1,
            'risk': 1 - lave_score,
        }

        questions_answered += recoverr_rollouts[qid]['question_answered']
        total_risk += 1-lave_score
        qids_completed += 1
        if qids_completed == args.num_examples:
            break
        json.dump(recoverr_rollouts, open(output_file, 'w'), indent=2)
        wandb_logger.log({
            'qid': qid,
            'num_samples': qids_completed,
            'total_cost': openai_caller.compute_cost(),
            'time_per_sample': time.time() - start_time,
            'coverage': 100.0*questions_answered/qids_completed,
            'risk': 0 if questions_answered == 0 else 100.0*total_risk/questions_answered,
        })
        t.set_description(f"Evaluating {dataset.name} (coverage = {questions_answered/qids_completed:.2%}, risk = {0 if questions_answered == 0 else total_risk/questions_answered:.2%}, cost=${openai_caller.compute_cost():.2f})")

    final_score = 100.0*eval_score/qids_completed
    print("{} score = {:.2f}%".format(dataset.name, final_score))
    json.dump(recoverr_rollouts, open(output_file, 'w'), indent=2)
    print("Total cost = ${:.2f}".format(openai_caller.compute_cost()))
    pdb.set_trace()

    ######### EVALUATION #########

    # Step 1: Load DirectVQA rollouts
    directvqa_experiment_dir = os.path.join(args.experiments_dir, f'{args.dataset}_{args.task_type}/{args.split}_outputs')
    directvqa_experiment_name = vlm_config['model_shorthand'].replace('_', '') + '_direct_vqa'
    directvqa_experiment_name += '-{}rollouts'.format(args.num_rollouts)
    directvqa_experiment_name += '-{}examples'.format(args.num_examples)
    directvqa_fn = os.path.join(directvqa_experiment_dir, f'{directvqa_experiment_name}.json')
    print(f"Loading directvqa outputs from {directvqa_fn}")
    is_correct_fn = input("If this is NOT the correct filename, press N, else press any key to continue.")
    if is_correct_fn.lower() == 'n':
        while True:
            directvqa_fn = input("Enter the correct filename: ")
            directvqa_fn = directvqa_fn.strip()
            if os.path.exists(directvqa_fn):
                break
            print(f"File {directvqa_fn} does not exist. Please try again.")
    directvqa_rollouts = json.load(open(directvqa_fn))
    all_qids = list(recoverr_rollouts.keys())
    directvqa_rollouts = {k: v for k, v in directvqa_rollouts.items() if k in all_qids}
    print(f"Loaded {len(directvqa_rollouts)} directvqa rollouts from {directvqa_fn}.")

    directselected_rollouts = []
    caption_rollouts = []
    recoverred_rollouts = []
    found_evidence_rollouts = []
    abstained_rollouts = []
    failed_to_recoverr_rollouts = []
    true_negative_rollouts = []
    total_goodanswers, vanilla_goodanswers_selected = 0, 0

    for qid in all_qids:
        r = recoverr_rollouts[qid]
        selected_evidences = r['selected_evidences']
        predicted_answer = r['answer']
        prediction_type = r['prediction_type']

        if directvqa_rollouts[str(r['qid'])][0]['lave_score'] == 1:
            total_goodanswers += 1
        
        if prediction_type == 'directvqa':
            directselected_rollouts.append(r)
            if directvqa_rollouts[str(r['qid'])][0]['lave_score'] == 1:
                vanilla_goodanswers_selected += 1
        elif prediction_type.startswith('recoverr'):
            recoverred_rollouts.append(r)
            #if len(selected_evidences) == 3:
            if prediction_type == 'recoverr_verbalizations':
                caption_rollouts.append(r)
            else:
                found_evidence_rollouts.append(r)
        else:
            abstained_rollouts.append(r)
            if directvqa_rollouts[str(r['qid'])][0]['lave_score'] == 1:
                failed_to_recoverr_rollouts.append(r)
            if directvqa_rollouts[str(r['qid'])][0]['lave_score'] == 0 and len(r['selected_evidences']) != 3:
                true_negative_rollouts.append(r)

    print(f"VLM produced {total_goodanswers} correct answers")

    def get_accuracies(rollouts, score_type):
        scores = [x[score_type] for x in rollouts]
        total_score = sum(scores)
        num_successful_rollouts = len(rollouts) - scores.count(0.0)
        directvqa_scores = [directvqa_rollouts[str(r['qid'])][0][score_type] for r in rollouts if str(r['qid']) in directvqa_rollouts]
        num_rollouts = len(rollouts)
        return total_score/num_rollouts, num_successful_rollouts/num_rollouts, sum(directvqa_scores)/num_rollouts

    def get_risk(rollouts):
        total_risk = sum([1 - p['lave_score'] for p in rollouts])/len(rollouts)
        return total_risk

    def get_effective_reliability(rollouts, dataset_size, error_cost):
        eff_rel = sum([p['lave_score'] if p['lave_score'] > 0 else -error_cost for p in rollouts])/dataset_size
        return eff_rel

    print(f"Number of abstained rollouts: {len(abstained_rollouts)} ({len(abstained_rollouts)/len(all_qids):.2%} of all rollouts)")
    if len(abstained_rollouts) != 0:
        acc, perc_successful, directvqa_score = get_accuracies(abstained_rollouts, 'score')
        lave_score, perc_lave_successful, directvqa_lavescore = get_accuracies(abstained_rollouts, 'lave_score')
        print(f"VQAScore Accuracy / LAVE score of abstained_rollouts: {acc:.2%} / {lave_score:.2%}")
        print(f"DirectVQA accuracy / LAVEscore of abstained_rollouts' A-OKVQA questions: {directvqa_score:.2%} / {directvqa_lavescore:.2%}")

    print(f"\nNumber of questions that were initially answered confidently: {len(directselected_rollouts)} ({len(directselected_rollouts)/len(all_qids):.2%} of all rollouts)")
    acc, perc_successful, directvqa_score = get_accuracies(directselected_rollouts, 'score')
    lave_score, perc_lave_successful, directvqa_lavescore = get_accuracies(directselected_rollouts, 'lave_score')
    print(f"VQAScore Accuracy / LAVE score of directselected_rollouts: {acc:.2%} / {lave_score:.2%}")
    #print(f"DirectVQA accuracy / LAVEscore of directselected_rollouts' A-OKVQA questions: {directvqa_score:.2%} / {directvqa_lavescore:.2%}")
    total_risk = get_risk(directselected_rollouts)
    eff_rel = get_effective_reliability(directselected_rollouts, len(all_qids), 1)
    print(f"Risk of selected subset: {total_risk:.2%}")
    print(f"Effective reliability without ReCoVERR: {eff_rel:.2%}")
    selected_good_directvqa_rollouts = len([r for r in directselected_rollouts if r['lave_score'] == 1])
    print(f"In initial selective prediction, {vanilla_goodanswers_selected} correct answers were actually answered (recall={vanilla_goodanswers_selected/total_goodanswers:.2%})")

    print(f"\nNumber of rollouts that ReCoVERR-ed evidence: {len(recoverred_rollouts)} ({len(recoverred_rollouts)/(len(all_qids) - len(directselected_rollouts)):.2%} of previously-abstained rollouts)")
    acc, perc_successful, directvqa_score = get_accuracies(recoverred_rollouts, 'score')
    lave_score, perc_lave_successful, directvqa_lavescore = get_accuracies(recoverred_rollouts, 'lave_score')
    print(f"VQAScore Accuracy / LAVE score of ReCoVERR-ed rollouts: {acc:.2%} / {lave_score:.2%}")
    #print(f"DirectVQA accuracy / LAVEscore of newly_selected_rollouts' A-OKVQA questions: {directvqa_score:.2%} / {directvqa_lavescore:.2%}")
    total_risk = get_risk(recoverred_rollouts)
    print(f"Risk of selected subset: {total_risk:.2%}")

    print(f"\nOut of ReCoVERRed rollouts,")
    if len(caption_rollouts) > 0:
        print(f"\tNumber of rollouts that answered based on caption+objects: {len(caption_rollouts)} ({len(caption_rollouts)/len(recoverred_rollouts):.2%} of ReCoVERR-ed rollouts)")
        total_risk = get_risk(caption_rollouts)
        print(f"\tRisk of caption_only subset: {total_risk:.2%}")

    if len(found_evidence_rollouts) > 0:
        print(f"\tNumber of rollouts that found some evidence: {len(found_evidence_rollouts)} ({len(found_evidence_rollouts)/len(recoverred_rollouts):.2%} of ReCoVERR-ed rollouts)")
        total_risk = get_risk(found_evidence_rollouts)
        print(f"\tRisk of found_evidence subset: {total_risk:.2%}")

    print(f"\nCombining initially answered and newly ReCoVERR-ed rollouts:")
    print(f"Coverage = {(len(directselected_rollouts) + len(recoverred_rollouts))/len(all_qids):.2%}")
    total_risk = get_risk(directselected_rollouts + recoverred_rollouts)
    print(f"Risk = {total_risk:.2%}")
    eff_rel = get_effective_reliability(directselected_rollouts + recoverred_rollouts, len(all_qids), 1)
    print(f"Effective reliability with ReCoVERR: {eff_rel:.2%}")
    recoverred_goodanswers = len([r for r in recoverred_rollouts if r['lave_score'] == 1])
    total_goodanswers_selected = recoverred_goodanswers + vanilla_goodanswers_selected
    recall = total_goodanswers_selected/total_goodanswers
    print(f"{total_goodanswers_selected} correct answers were actually answered (recall={recall:.2%})")    

    num_good_recoverred = len([r for r in recoverred_rollouts if r['lave_score'] == 1])
    print(f"\nReCoVERR-ed {num_good_recoverred} good answers")
    print(f"Failed to ReCoVERR {len(failed_to_recoverr_rollouts)} good answers")
    print(f"Successfully did not verify {len(true_negative_rollouts)} bad answers")
    recoverry_rate = num_good_recoverred/(num_good_recoverred + len(failed_to_recoverr_rollouts))
    print(f"Percentage of good answers ReCoVeRR-ed: {recoverry_rate:.2%}")

if __name__ == '__main__':
    main()