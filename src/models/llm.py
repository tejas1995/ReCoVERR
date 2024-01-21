import pdb

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn

from utils.mem_utils import calculate_model_size_gb

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class T5LLM:
    
    def __init__(self, config, device):

        self.config = config
        self.device = device
        model_name = config['model_name']
        display_name = config['display_name']
        self.rephrasing_inference_params = config['rephrasing_inference_params']
        self.rephrasing_examples = config['rephrasing_examples']
        #model_name = 'google/t5_xxl_true_nli_mixture'

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        yn_tokens = self.tokenizer.tokenize("yes no")
        self.yn_token_ids = self.tokenizer.convert_tokens_to_ids(yn_tokens)
        logger.info(f"Loaded {display_name} language model")
        logger.info(f"Model size: {calculate_model_size_gb(self.model):.2f}GB")
        logger.info("-"*100)

    def get_batched_entailment_confidences(self, premises, hypotheses):
        prompts = [f"Premise: {p} \n\n" \
            f"Hypothesis: {h} \n" \
            f"Can we infer the hypothesis from the premise? Options: yes, no. Answer: " for p, h in zip(premises, hypotheses)]

        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        output_ids = self.tokenizer([""]*len(prompts), return_tensors='pt', truncation=True).input_ids.to(self.device)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=output_ids)
        
        logits = output.logits[:, 0]
        probs = nn.Softmax(dim=-1)(logits)
        confidences = []
        for i in range(len(prompts)):
            yes_prob = probs[i, self.yn_token_ids[0]].item()
            no_prob = probs[i, self.yn_token_ids[1]].item()
            conf = yes_prob/(yes_prob+no_prob)
            confidences.append(conf)
        return confidences

    def get_entailment_confidence(self, premise, hypothesis):
        prompt = f"Premise: {premise} \n" \
            f"Hypothesis: {hypothesis} \n\n" \
            f"Can we infer the hypothesis from the premise? Options: yes, no. Answer: "

        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        output_ids = self.tokenizer("", return_tensors='pt', truncation=True).input_ids.to(self.device)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=output_ids)
        
        logits = output.logits[:, 0]
        probs = nn.Softmax(dim=-1)(logits)
        yes_prob = probs[0, self.yn_token_ids[0]].item()
        no_prob = probs[0, self.yn_token_ids[1]].item()
        conf = yes_prob/(yes_prob+no_prob)
        #pdb.set_trace()
        return conf

    def rephrase_qa_to_statement(self, question, answer):
        prompt = f"Rephrase the question and answer into a single statement. " \
            "The re-phrased statement should summarize the question and answer. " \
            "The re-phrased statement should not be a question. \n " + \
            '\n\n'.join(self.rephrasing_examples) + \
            f" \n\nQuestion: {question} \n " \
            f"Answer: {answer} \n " \
            "Statement: "
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
        output_ids = self.model.generate(input_ids, **self.rephrasing_inference_params)
        rephrased_question = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        #pdb.set_trace()
        if '?' in rephrased_question:
            return f"{question} {answer}."
        else:
            return rephrased_question
    
    def rephrase_batched_qas_to_statements(self, questions, answers):
        prompts = [f"Rephrase the question and answer into a single statement. " \
            "The re-phrased statement should summarize the question and answer. " \
            "The re-phrased statement should not be a question. \n " + \
            '\n\n'.join(self.rephrasing_examples) + \
            f"Question: {q} \n" \
            f"Answer: {a} \n" \
            "Statement: " for q, a in zip(questions, answers)]
        input_ids = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).input_ids.to(self.device)
        output_ids = self.model.generate(input_ids, **self.rephrasing_inference_params)
        rephrased_questions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        #for q, a, r in zip(questions, answers, rephrased_questions):
        #    logger.info(f"Question: {q}")
        #    logger.info(f"Answer: {a}")
        #    logger.info(f"Rephrased question: {r}")
        #    logger.info("-"*100)
        #pdb.set_trace()
        for i, r in enumerate(rephrased_questions):
            if '?' in r:
                rephrased_questions[i] = f"{questions[i]} {answers[i]}."
        return rephrased_questions
    
LLM_CLASS_MAP = {
    't5': T5LLM
}