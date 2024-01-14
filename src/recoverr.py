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

def update_state_with_evidences(state, evidences):
    copy_state = copy.deepcopy(state)
    step_num = len(state.step2interaction)
    for evidence in evidences:
        copy_state.update_state(
                    step_num=step_num,
                    action=evidence['question'], 
                    new_result=evidence['answer'],
                    result_conf=evidence['vlm_conf']
                )
        step_num += 1
    return copy_state

def run_recoverr_verifyingevidences(
    query_details, 
    qgen_model, 
    vlm_model, 
    llm_model, 
    recoverr_config, 
    do_print=False
) -> Dict:

    max_evidence_collection_turns = recoverr_config['max_evidence_collection_turns']
    questions_generated_per_turn = recoverr_config['questions_generated_per_turn']
    min_nliconf_threshold = recoverr_config['min_nliconf_threshold']
    desired_risk = recoverr_config['desired_risk']
    guessconf_minimum_defeasibility_delta = recoverr_config['guessconf_minimum_defeasibility_delta']
    image_aware_guesser = recoverr_config['image_aware_guesser']

    #region DIRECTVQA    
    directvqa_predicted_answer, directvqa_answer_logprobsdict = vlm_model.ask(
            raw_image=query_details['image'], 
            question=query_details['question']
        )
    directvqa_conf = directvqa_answer_logprobsdict['yn_prob']
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
    #endregion

    #region ANSWER USING CAPTION ONLY
    if image_aware_guesser is False:
        caption, caption_logprobs_dict = vlm_model.caption(query_details['image'])
        caption_conf = caption_logprobs_dict['yn_prob']
        caption_nli_conf = llm_model.get_entailment_confidence(
            premise=caption,
            hypothesis=directvqa_hypothesis
        )
        if do_print:
            print(f"Image caption: {caption} (confidence={caption_conf:.4f})")
            print(f"Entailment likelihood that DirectVQA answer is '{directvqa_predicted_answer}': {caption_nli_conf:.4f}")
        caption_evidence = {
                    'question': 'Describe the image',
                    'answer': caption,
                    'vlm_conf': caption_conf,
                    'statement': caption, 
                    'is_reliable': True if caption_conf >= 1-desired_risk else False,
                    'is_relevant': True,
                    'prediction_entailment_conf': caption_nli_conf,
        }        
        expected_conf = caption_conf*caption_nli_conf
        if caption_nli_conf >= min_nliconf_threshold:
        #if expected_conf >= 1-desired_risk:
            if do_print:
                print(f"Based on caption, predicting '{directvqa_predicted_answer}' with confidence {caption_nli_conf:.4f}")
            return {
                'prediction': directvqa_predicted_answer,
                'prediction_entailment_conf': caption_nli_conf,
                'visual_conf': caption_conf, 
                'overall_conf': expected_conf, 
                'all_evidences': [caption_evidence], 
                'reliable_evidences': [caption_evidence], 
                'reliable_and_relevant_evidences': [caption_evidence],
            }
    #endregion

    base_evidences = [] if image_aware_guesser else [caption_evidence]    
    all_evidences = [caption_evidence]
    reliable_evidences = []
    reliable_and_relevant_evidences = []
    for j in range(max_evidence_collection_turns):
        #region FIND_RELIABLE_EVIDENCES
        # Gather different evidences about the image, retain only the reliable ones
        if do_print:
            print(f"{'-'*100}\nEvidence collection turn {j}")
        candidate_questions = qgen_model.generate_supportingevidence_questions(
            target_question=query_details['question'], 
            evidences=[caption_evidence]+reliable_evidences, 
            num_questions=questions_generated_per_turn, 
            possible_answer=directvqa_predicted_answer
        )
        candidate_vqa_answers, vqa_answer_logprobsdicts = vlm_model.ask_multiplequestions(
            raw_image=query_details['image'],
            questions=candidate_questions
        )
        candidate_vqa_confidences = [x['yn_prob'] for x in vqa_answer_logprobsdicts]
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
                    'is_reliable': True if candidate_vqa_confidences[i] >= 1-desired_risk else False, 
                    'is_relevant': False,
                }

            # Only retain a VQA evidence if it is "reliable"
            if candidate_vqa_confidences[i] >= 1-desired_risk:
                # Check if evidence is also relevant
                if guessconf_minimum_defeasibility_delta > 0:
                    # Answer question based solely on basis of collected evidence
                    #true_evidence_premise = f"{caption} {evidence['statement']}"
                    true_evidence_premise = evidence['statement']
                    true_evidence_nliconf = llm_model.get_entailment_confidence(
                        premise=true_evidence_premise,
                        hypothesis=directvqa_hypothesis
                    )

                    # Estimate confidence of previous guess IF the evidence is false
                    #print("Counterfactual:")
                    false_evidence = copy.deepcopy(evidence)
                    false_evidence['answer'] = f"not {false_evidence['answer']}"
                    false_evidence['statement'] = llm_model.rephrase_qa_to_statement(
                        question=false_evidence['question'],
                        answer=false_evidence['answer']
                    )
                    #false_evidence_premise = f"{caption} {false_evidence['statement']}"
                    false_evidence_premise = false_evidence['statement']
                    false_evidence_nliconf = llm_model.get_entailment_confidence(
                        premise=false_evidence_premise,
                        hypothesis=directvqa_hypothesis
                    )

                    evidence['prediction_entailment_conf'] = true_evidence_nliconf
                    evidence['counterfactual_prediction_entailment_conf'] = false_evidence_nliconf

                    if do_print:
                        print(f"P({directvqa_hypothesis} | {true_evidence_premise}) = {true_evidence_nliconf:.4f}")
                        print(f"P({directvqa_hypothesis} | {false_evidence_premise}) = {false_evidence_nliconf:.4f}")    
                    #if agent_guess_conf > min_nliconf_threshold and \
                    if (np.abs(true_evidence_nliconf - false_evidence_nliconf) > guessconf_minimum_defeasibility_delta):
                        reliable_and_relevant_evidences.append(evidence)
                        evidence['is_relevant'] = True
                        if do_print:
                            print("RELEVANT")
                    else:
                        if do_print:
                            print("NOT RELEVANT")
                        pass
                    #print()
                else:
                    if do_print:
                        print("RELEVANT")
                    evidence['is_relevant'] = True
                    reliable_and_relevant_evidences.append(evidence)
                reliable_evidences.append(evidence)
            all_evidences.append(evidence)

        if do_print:
            print(f"After evidence collection turn {j}:")
            print(f"Reliable evidences: {len(reliable_evidences)}")
            print(f"Reliable and relevant evidences: {len(reliable_and_relevant_evidences)}")

        # Try to get confidence in original guess based on all reliable and relevant evidences so far
        if len(reliable_and_relevant_evidences) > 0:
            aggregated_evidences_premise = ' '.join([caption] + [x['statement'] for x in reliable_and_relevant_evidences])
            aggregated_evidences_nliconf = llm_model.get_entailment_confidence(
                premise=aggregated_evidences_premise,
                hypothesis=directvqa_hypothesis
            )
            if do_print:
                print(f"P({directvqa_hypothesis} | {aggregated_evidences_premise}) = {aggregated_evidences_nliconf:.4f}")

            min_evidence_conf = min([x['vlm_conf'] for x in reliable_and_relevant_evidences])
            expected_conf = min_evidence_conf*aggregated_evidences_nliconf
            if do_print:
                print(f"Min evidence conf = {min_evidence_conf:.4f}")
                print(f"Expected conf = {expected_conf:.4f}")
            #if expected_conf >= (1-desired_risk):
            if aggregated_evidences_nliconf >= min_nliconf_threshold:
                return {
                    'prediction': directvqa_predicted_answer,
                    'prediction_entailment_conf': aggregated_evidences_nliconf,
                    'overall_conf': expected_conf, 
                    'all_evidences': all_evidences, 
                    'reliable_evidences': reliable_evidences, 
                    'reliable_and_relevant_evidences': reliable_and_relevant_evidences,
                }



    return {
                'prediction': "unknown",
                'prediction_entailment_conf': 1.0,
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
    parser.add_argument("--do_in_context_learning", action='store_true')
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--directvqa_abstained_qids_file", type=str, default=None)
    parser.add_argument("--experiments_dir", type=str, default='/net/nfs.cirrascale/mosaic/tejass/experiments/recoverr')
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

    # Load question generation model
    qgen_class = config['qgen']['class_name']
    qgen_model_class = QGEN_CLASS_MAP[qgen_class]
    qgen_config = yaml.safe_load(open(config['qgen']['model_config_path']))
    qgen_model = qgen_model_class(qgen_config, device)

    # Load VLM
    vlm_class = config['vlm']['class_name']
    vlm_model_class = VLM_CLASS_MAP[vlm_class]
    vlm_config = yaml.safe_load(open(config['vlm']['model_config_path']))
    vlm_model = vlm_model_class(vlm_config, device)
    vlm_model.set_vqa_inference_params(config['vlm']['vqa_inference_params'])
    vlm_model.set_caption_inference_params(config['vlm']['caption_inference_params'])

    # Load LLM
    llm_class = config['llm']['class_name']
    llm_model_class = LLM_CLASS_MAP[llm_class]
    llm_config = yaml.safe_load(open(config['llm']['model_config_path']))
    llm_model = llm_model_class(llm_config, device)

    #directvqa_abstained_qids_file = f"/net/nfs.cirrascale/mosaic/tejass/data/directvqa_abstained_qids/{vqa_model.config['model_shorthand'].replace('_', '')}-{args.dataset}_{args.split}_{args.task_type}.txt"
    directvqa_abstained_qids_file = args.directvqa_abstained_qids_file
    if directvqa_abstained_qids_file and os.path.exists(directvqa_abstained_qids_file):
        directvqa_abstained_qids = set([int(qid.strip()) for qid in open(directvqa_abstained_qids_file).readlines()])
        args.num_examples = len(directvqa_abstained_qids)
    else:
        directvqa_abstained_qids = set([dataset.data[i]['qid'] for i in range(len(dataset))])
    logger.info(f"Number of DirectVQA-abstained questions that we are doing search for: {len(directvqa_abstained_qids)}")
    qid2idx = {str(dataset.data[i]['qid']): i for i in range(len(dataset))}
    directvqa_abstained_idxs = {qid2idx[str(qid)] for qid in directvqa_abstained_qids}

    # Create experiment directories
    recoverr_config = config['recoverr']
    experiment_name = config['exp_name']
    experiment_name += '-{}maxcollectionturns-{}questionsperturn'.format(
        recoverr_config['max_evidence_collection_turns'],
        recoverr_config['questions_generated_per_turn'],
    )
    experiment_name += '-{}desiredrisk-{}defeasibilitydelta-{}minguessconf'.format(
        recoverr_config['desired_risk'],
        recoverr_config['guessconf_minimum_defeasibility_delta'],
        recoverr_config['min_nliconf_threshold']
    )
    if args.num_rollouts != 1:
        experiment_name += '-{}rollouts'.format(args.num_rollouts)
    if args.num_examples != len(dataset):
        experiment_name += '-{}examples'.format(args.num_examples)
    experiment_dir = os.path.join(args.experiments_dir, f'{args.dataset}_{args.task_type}/{args.split}_outputs/recoverr_verifyingevidences_rollouts')

    wandb_logger.initialize(
        wandb_config_filename=args.wandb_config_file, 
        experiment_name=f'{args.dataset}_{args.split}-{experiment_name}',
        #project_name='beamsearch_vqa'
        project_name='recoverr_verifyingevidences'
    )
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    output_file = f'{experiment_dir}/{experiment_name}.json'
    logger.info(f"Output file: {output_file}")
    #pdb.set_trace()

    eval_score, total_lave_score = 0.0, 0.0
    #t = tqdm(range(len(dataset)))
    t = tqdm(directvqa_abstained_idxs)
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
    #pdb.set_trace()
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
            result_dict = run_recoverr_verifyingevidences(
                query_details=query_details,
                qgen_model=qgen_model,
                llm_model=llm_model,
                vlm_model=vlm_model,
                recoverr_config=recoverr_config,
                do_print=False
            )
            predicted_answer = result_dict['prediction']
            if args.dataset == 'okvqa':
                predicted_answer = postprocess_ok_vqa_generation(lemmatize(predicted_answer))
            prediction_entailment_conf = result_dict['prediction_entailment_conf']
            overall_conf = result_dict['overall_conf']
            gathered_evidences = result_dict['all_evidences']
            selected_evidences = result_dict['reliable_and_relevant_evidences']
            #pdb.set_trace()

            #answer_prediction_history = terminal_state.answer_prediction_history
            score = score_dict[predicted_answer]

            #answer_found = determine_if_answer_found(question, answer)
            answer_found = True
            lave_reasoning, lave_score = lave_scorer.compute(
                prediction=predicted_answer,
                references=data['reference_answers'],
                question=question,
            )

            qid2rollouts[qid].append(
                    {'qid': qid,
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
                    'prediction_entailment_conf': prediction_entailment_conf,
                    'overall_conf': overall_conf,
                    }
                )

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