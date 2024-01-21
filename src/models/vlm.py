import pdb
from typing import List
import pickle as pkl

import numpy as np
import torch
from torch import nn
from lavis.models import load_model_and_preprocess, load_preprocess
from lavis.common.registry import registry
from omegaconf import OmegaConf

from utils.mem_utils import calculate_model_size_gb

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

def load_model_with_custom_configs(name, config, is_eval=False, device="cpu"):
    model_cls = registry.get_model_class(name)
    cfg = OmegaConf.create(config)
    if 'model_config' not in config:
        model, _, _ = load_model_and_preprocess(
            name=config['name'],
            model_type=config['type'],
            is_eval=is_eval,
            device=device,
        )
    else:
        model = model_cls.from_config(cfg.model_config)

    preprocess_cfg = cfg.preprocess_config
    vis_processors, txt_processors = load_preprocess(preprocess_cfg)

    if is_eval:
        model.eval()

    return model.to(device), vis_processors, txt_processors


class BLIP(nn.Module):
    def __init__(self, config, device):
        super(BLIP, self).__init__()

        self.config = config
        self.model_name = config['class_name']
        self.display_name = config['model_display_name']
        load_config = config["pt_model_load"]
        self.model, self.vis_processors, self.text_processors = load_model_with_custom_configs(
                name=load_config['name'],
                config=load_config,
                device=device
            )
        self.vqa_inference_params = {}
        yn_tokens = self.model.t5_tokenizer.tokenize("yes no")
        self.yn_token_ids = self.model.t5_tokenizer.convert_tokens_to_ids(yn_tokens)

        #preproc_config = OmegaConf.create(config['preprocess_config'])
        #self.vis_processors, self.text_processors = load_preprocess(preproc_config)
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad is True)

        self.device = device
        self.use_confidence_calibrator = False
        logger.info(f"Loaded BLIP (model={self.display_name})!")
        logger.info("Model size: {:.2f}B parameters ({:.2f}B trainable), {:.2f}GB in memory".format(num_params*10**-9, num_trainable_params*10**-9, calculate_model_size_gb(self.model)))

    def preprocess_batch(self, batch, mode='train'):
        images = batch['images']
        text_inputs = batch['text_inputs']
        text_outputs = batch['text_outputs']
        
        processed_images = torch.stack([self.vis_processors[mode](img) for img in images], dim=0).to(self.device)
        processed_text_inputs = [self.text_processors[mode](txt) for txt in text_inputs]
        processed_text_outputs = [self.text_processors[mode](txt) for txt in text_outputs]
        samples = {
            "image": processed_images,
            "text_input": processed_text_inputs,
            "text_output": processed_text_outputs,
            "prompt": text_inputs,
            "score_dict": batch['score_dict'],
            "qids": batch['qids'],
        }
        return samples

    def set_vqa_inference_params(self, vqa_inference_params):
        self.vqa_inference_params = vqa_inference_params

    def set_caption_inference_params(self, caption_inference_params):
        self.caption_inference_params = caption_inference_params
    
    def set_confidence_calibrator(self, calibrator_model_path):
        self.use_confidence_calibrator = True
        self.confidence_calibrator = pkl.load(open(calibrator_model_path, "rb"))
        logger.info(f"Loaded VLM confidence calibrator from {calibrator_model_path}!")

    def forward(self, batch):
        output_dict = self.model(batch)
        loss = output_dict["loss"]
        return loss

    def generate(self, batch):
        with torch.no_grad():
            output = self.model.generate(batch, **self.vqa_inference_params)
        output = [x.lower() for x in output]
        return output

    def get_answer_confidence(self, raw_image, questions, answers):
        prompts = [f"Question: {q} \n" \
            f"Answer: {a} \n" \
            f"Is the given answer correct for the question? Answer yes or no: " for q, a in zip(questions, answers)]
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        image = image.repeat(len(questions), 1, 1, 1)
        output = self.model({"image": image, "text_input": prompts, "text_output": ["yes"]*len(questions)})
        logits = output["outputs"]["logits"][:, 0, :]

        yes_logits = logits[:, self.yn_token_ids[0]]
        no_logits = logits[:, self.yn_token_ids[1]]
        yn_logits = torch.cat([yes_logits.unsqueeze(1), no_logits.unsqueeze(1)], dim=1)
        if self.use_confidence_calibrator:
            yn_logits = yn_logits.cpu().detach().to(torch.float32).numpy()
            yn_probs = self.confidence_calibrator.predict_proba(yn_logits)[:, 1]
        else:
            yn_probs = nn.Softmax(dim=-1)(yn_logits)[:, 0]
        #print(f"yes_prob: {yes_prob}, no_prob: {no_prob}, sum: {yes_prob + no_prob}")
        return yn_probs.tolist(), yn_logits.tolist()

    def get_vqa_pred(self, question, raw_image):
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        prompt = f"Question: {question} Short answer: "
        answer = self.model.generate({"image": image, "prompt": prompt}, length_penalty=-1.0)[0].lower()
        return answer

    def ask(self, raw_image, question, **kwargs):
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        prompt = f"Question: {question} Short answer:"
        output = self.model.generate({"image": image, "prompt": prompt}, **self.vqa_inference_params, **kwargs)
        if len(output) == 2:
            answer = output[0][0].lower()
            outputs = output[1]
            transition_scores = self.model.t5_model.compute_transition_scores(
                outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
                #outputs.sequences, outputs.scores, normalize_logits=True
            )
            generated_ids = outputs.sequences[:, 1:]
            answer, logprobs_dict = self.get_answer_with_probability(
                    transition_scores[0].tolist(), 
                    generated_ids[0].tolist(), 
                )
            yn_probs, yn_logits = self.get_answer_confidence(raw_image, [question], [answer])
            logprobs_dict["yn_prob"] = yn_probs[0]
            logprobs_dict["yn_logits"] = yn_logits[0]
            return answer.lower(), logprobs_dict

        else:
            answer = output[0].lower()
            return answer, torch.FloatTensor([0.0])

    def caption(self, raw_image, prompt="A photo of", **kwargs):
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image, "prompt": prompt}, **self.caption_inference_params, **kwargs)
        if len(output) == 2:
            answer = output[0][0].lower()
            outputs = output[1]
            transition_scores = self.model.t5_model.compute_transition_scores(
                outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
                #outputs.sequences, outputs.scores, normalize_logits=True
            )
            generated_ids = outputs.sequences[:, 1:]
            answer, logprobs_dict = self.get_answer_with_probability(
                    transition_scores[0].tolist(), 
                    generated_ids[0].tolist(), 
                )
            yn_probs, yn_logits = self.get_answer_confidence(raw_image, [prompt], [answer])
            logprobs_dict["yn_prob"] = yn_probs[0]
            logprobs_dict["yn_logits"] = yn_logits[0]#.tolist()
            return f"An image of {answer.lower()}.", logprobs_dict

        else:
            answer = output[0].lower()
            return answer, torch.FloatTensor([0.0])

    def ask_multiplequestions(self, raw_image, questions: List[str], **kwargs):
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        image = image.repeat(len(questions), 1, 1, 1)
        prompts =[f"Question: {question} Short answer: " for question in questions]
        output = self.model.generate({"image": image, "prompt": prompts}, **self.vqa_inference_params, **kwargs)

        if len(output) == 2:
            outputs = output[1]
            transition_scores = self.model.t5_model.compute_transition_scores(
                outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
                #outputs.sequences, outputs.scores, normalize_logits=True
            )
            tokenizer = self.model.t5_tokenizer
            generated_ids = outputs.sequences[:, 1:]

            num_questions = len(questions)
            sep_token_id  = tokenizer.convert_tokens_to_ids('</s>')
            pad_token_id = tokenizer.pad_token_id
            logprobs_dicts = []
            answers = []
            for i in range(len(generated_ids)):
                answer, logprobs_dict = self.get_answer_with_probability(
                    transition_scores[i].tolist(), 
                    generated_ids[i].tolist(), 
                )
                answers.append(answer)
                logprobs_dicts.append(logprobs_dict)

            yn_probs, yn_logits = self.get_answer_confidence(raw_image, questions, answers)
            for i, yn_prob in enumerate(yn_probs):
                logprobs_dicts[i]["yn_prob"] = yn_prob
                logprobs_dicts[i]["yn_logits"] = yn_logits
            confs = [logprobs_dict["yn_prob"] for logprobs_dict in logprobs_dicts]
            return answers, logprobs_dicts

    def ask_questions_batched(self, raw_images: List, questions: List[str], **kwargs):
        assert len(raw_images) == len(questions)
        procd_images = []
        for raw_image in raw_images:
            procd_images.append(self.vis_processors["eval"](raw_image))
        images = torch.stack(procd_images).to(self.device)
        prompts =[f"Question: {question} Short answer: " for question in questions]
        output = self.model.generate({"image": images, "prompt": prompts}, **self.vqa_inference_params, **kwargs)

        if len(output) == 2:
            outputs = output[1]
            transition_scores = self.model.t5_model.compute_transition_scores(
                outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
                #outputs.sequences, outputs.scores, normalize_logits=True
            )
            tokenizer = self.model.t5_tokenizer
            generated_ids = outputs.sequences[:, 1:]

            num_questions = len(questions)
            sep_token_id  = tokenizer.convert_tokens_to_ids('</s>')
            pad_token_id = tokenizer.pad_token_id
            logprobs_dicts = []
            answers = []
            for i in range(len(generated_ids)):
                answer, logprobs_dict = self.get_answer_with_probability(
                    transition_scores[i].tolist(), 
                    generated_ids[i].tolist(), 
                )
                answers.append(answer)
                logprobs_dicts.append(logprobs_dict)

            yn_probs, yn_logits = self.get_answer_confidence(raw_image, questions, answers)
            for i, yn_prob in enumerate(yn_probs):
                logprobs_dicts[i]["yn_prob"] = yn_prob
                logprobs_dicts[i]["yn_logits"] = yn_logits
            confs = [logprobs_dict["yn_prob"] for logprobs_dict in logprobs_dicts]
            return answers, logprobs_dicts

    def ask_multiplequestions_samplemultianswers(self, raw_image, questions: List[str], num_answers, **kwargs):
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        image = image.repeat(len(questions), 1, 1, 1)
        prompts =[f"Question: {question} Short answer: " for question in questions]
        output = self.model.generate({"image": image, "prompt": prompts}, num_captions=num_answers, **self.vqa_inference_params, **kwargs)

        if len(output) == 2:
            outputs = output[1]
            transition_scores = self.model.t5_model.compute_transition_scores(
                outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
                #outputs.sequences, outputs.scores, normalize_logits=True
            )
            tokenizer = self.model.t5_tokenizer
            generated_ids = outputs.sequences[:, 1:]

            num_questions = len(questions)
            sep_token_id  = tokenizer.convert_tokens_to_ids('</s>')
            pad_token_id = tokenizer.pad_token_id
            logprobs_dicts = []
            answers = []
            for i in range(len(generated_ids)):
                answer, logprobs_dict = self.get_answer_with_probability(
                    transition_scores[i].tolist(), 
                    generated_ids[i].tolist(), 
                )
                answers.append(answer)
                logprobs_dicts.append(logprobs_dict)
            pdb.set_trace()

            yn_probs = self.get_answer_confidence(raw_image, questions, answers)
            for i, yn_prob in enumerate(yn_probs):
                logprobs_dicts[i]["yn_prob"] = yn_prob
            return answers, logprobs_dicts

    def ask_multianswers(self, raw_image, question, num_answers, **kwargs):
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        prompt = question
        output = self.model.generate({"image": image, "prompt": prompt}, num_captions=num_answers, **self.vqa_inference_params, **kwargs)
        if len(output) == 2:
            answers = [x.lower() for x in output[0]]
            answer_logprobs = [output[1][i].item() for i in range(num_answers)]
        else:
            answers = [x.lower() for x in output]
            answer_logprobs = [0.0]*len(answers)
        return answers, answer_logprobs

    def get_answer_with_probability(self, scores, gen_ids):
        pad_token_id = self.model.t5_tokenizer.pad_token_id
        filtered_tokens = [(x, y) for x, y in zip(gen_ids, scores) if x != pad_token_id]        # Remove pad tokens
        gen_ids = [x[0] for x in filtered_tokens]
        answer = self.model.t5_tokenizer.decode(gen_ids, skip_special_tokens=True)    
        answer_tokens = self.model.t5_tokenizer.convert_ids_to_tokens(gen_ids)
        answer_token_logprobs = [x[1] for x in filtered_tokens]
        if len(answer_tokens) == 1:      # SEP token only
            answer_tokens = ["", "[SEP]"]
            answer_token_logprobs = [-np.inf, 0.0]

        first_token_logprob = answer_token_logprobs[0]
        first_token_prob = np.exp(first_token_logprob)
        min_token_logprob =  min(answer_token_logprobs)
        min_token_prob = np.exp(min_token_logprob)
        mean_token_logprob = sum(answer_token_logprobs[:-1]) / (len(answer_token_logprobs) - 1)
        exp_mean_token_logprob = np.exp(mean_token_logprob)
        mean_token_prob = sum(np.exp(answer_token_logprobs[:-1])) / (len(answer_token_logprobs) - 1)
        token_probs = np.exp(answer_token_logprobs).tolist()
        prod_token_probs = np.prod(np.exp(answer_token_logprobs))
        logprobs_dict = {
            "first_token_logprob": first_token_logprob,
            "first_token_prob": first_token_prob,
            "min_token_logprob": min_token_logprob,
            "min_token_prob": min_token_prob,
            "mean_token_logprob": mean_token_logprob,
            "mean_token_prob": mean_token_prob,
            "exp_mean_token_logprob": exp_mean_token_logprob, 
            "answer_token_logprobs": answer_token_logprobs,
            "token_probs": token_probs,
            "answer_tokens": answer_tokens,
            "prod_token_probs": prod_token_probs
        }
        return answer, logprobs_dict

VLM_CLASS_MAP = {
    "blip": BLIP
}