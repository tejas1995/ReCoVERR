import pdb
import yaml
from typing import List, Union, Dict, Tuple
from PIL import Image
import re

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

from utils.openai_utils import openai_caller
from utils.mem_utils import calculate_model_size_gb

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

def add_all_special_tokens(tokenizer):
    """
        special_tokens_dict = {"cls_token": "<CLS>"}

        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print("We have added", num_added_toks, "tokens")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))

        assert tokenizer.cls_token == "<CLS>"

    """
    original_len: int = len(tokenizer)
    num_added_toks: dict = {}
    if tokenizer.pad_token is None:
        num_added_toks['pad_token'] = "<pad>"
    if tokenizer.cls_token is None:
        num_added_toks['cls_token'] = "<cls>"
    if tokenizer.sep_token is None:
        num_added_toks['sep_token'] = "<sep>"
    if tokenizer.mask_token is None:
        num_added_toks['mask_token'] = "<mask>"
    # num_added_toks = {"bos_token": "<bos>", "cls_token": "<cls>", "sep_token": "<s>", "mask_token": "<mask>"}
    # special_tokens_dict = {'additional_special_tokens': new_special_tokens + tokenizer.all_special_tokens}
    num_new_tokens: int = tokenizer.add_special_tokens(num_added_toks)
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.cls_token == "<cls>"
    assert tokenizer.sep_token == "<sep>"
    assert tokenizer.mask_token == "<mask>"
    msg = f"Error, not equal: {len(tokenizer)=}, {original_len + num_new_tokens=}"
    assert len(tokenizer) == original_len + num_new_tokens, msg



class GPTQuestionGenerator:

    def __init__(self, config: Dict, device: torch.device):
        #super().__init__(config, device)
        #device = torch.device("cuda:1")
        #device = torch.device("cpu")
        self.config = config
        self.device = device

        self.gpt_type = config['gpt_type']
        self.qgen_model_name = config['qgen_model_name']
        #self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
        self.max_new_question_tokens = config['max_new_question_tokens']
        self.question_gen_temperature = config['question_gen_temperature']

        self.questiongen_prompts = yaml.load(open(config['questiongen_prompts_file'], 'r'), Loader=yaml.FullLoader)

        logger.info(f"Initialized {self.qgen_model_name} QuestionGenerator")
        #logger.info("-"*100)

    def __str__(self) -> str:
        return f"Question Generator: {self.qgen_model_name}"

    def add_tokens(self, new_tokens: List):
        self.tokenizer.add_tokens(new_tokens)
        logger.info(f"Added new tokens to Agent: {', '.join(new_tokens)}")

    def post_process_gptresponse(self, gpt_response):
        try:
            lines = [x for x in gpt_response.split('\n') if len(x) > 0]
            command = lines[0].strip()
        except:
            #pdb.set_trace()
            command = '<vqa>(question="Error encountered.", image=IMG_0)'
        return command

    def post_process_gptanswer(self, gpt_answer):
        answer = gpt_answer.lower()
        if "answer:" in answer:
            answer = answer.split("answer:")[-1].strip()
        return answer

    def remove_question_number(self, question):
        return re.sub(r'[0-9]+\.', '', question).strip()

    def generate_multiple_nextquestions(self, target_question: str, evidences: List[Dict], num_questions: int = 1) -> str:
        init_sys_prompt = {
            'role': 'system', 
            'content': self.questiongen_prompts['init_sys_prompt']
        }
        final_sys_prompt = self.questiongen_prompts['final_sys_prompt'].replace('TARGET_QUESTION', target_question).replace('NUM_QUESTIONS', str(num_questions))
        final_sys_prompt = {
            'role': 'system', 
            'content': final_sys_prompt
        }

        input_messages = [init_sys_prompt]
        for e in evidences:
            input_messages.append({
                'role': 'user', 
                'content': e['sentence']
            })
        input_messages += [final_sys_prompt]

        gpt_response = openai_caller(
                input_messages, 
                model=self.gpt_type, 
                max_new_tokens=self.max_new_question_tokens, 
                temperature=self.question_gen_temperature,
                num_completions=1
            )
        questions = [x.strip() for x in gpt_response.split('\n')]
        questions = [q for q in questions if len(q) > 0]
        questions = [self.remove_question_number(q) for q in questions]
        return questions

    def generate_supportingevidence_questions(
            self, 
            target_question: str, 
            evidences: List[Dict], 
            num_questions: int = 1, 
            possible_answer: str = None
        ) -> List[str]:

        init_sys_prompt = {
            'role': 'system', 
            'content': self.questiongen_prompts['init_sys_prompt']
        }
        final_sys_prompt = self.questiongen_prompts['final_sys_prompt']
        final_sys_prompt = final_sys_prompt.replace('TARGET_QUESTION', target_question)
        final_sys_prompt = final_sys_prompt.replace('NUM_QUESTIONS', str(num_questions))
        if possible_answer is not None:
            final_sys_prompt = final_sys_prompt.replace('POSSIBLE_ANSWER', possible_answer)
        final_sys_prompt = {
            'role': 'system', 
            'content': final_sys_prompt
        }

        input_messages = [init_sys_prompt]
        if len(evidences) > 0:
            input_messages.append({
                'role': 'user',
                'content': 'What you already know about the image:'
            })
            for e in evidences:
                input_messages.append({
                    'role': 'user', 
                    'content': e['statement']
                })
        input_messages.append(final_sys_prompt)

        gpt_response = openai_caller(
                input_messages, 
                model=self.gpt_type, 
                max_new_tokens=self.max_new_question_tokens, 
                temperature=self.question_gen_temperature,
                num_completions=1
            )
        questions = [x.strip() for x in gpt_response.split('\n')]
        questions = [q for q in questions if len(q) > 0]
        questions = [self.remove_question_number(q).strip() for q in questions]
        questions = [q for q in questions if q.endswith('?') and len(q) > 0]
        #if len(questions) == 0:
        #    pdb.set_trace()
        return questions

class MistralQuestionGenerator:

    def __init__(self, config: Dict, device: torch.device):
        #super().__init__(config, device)
        self.config = config
        self.device = device

        self.agent_name = config['agent_name']
        self.model = AutoModelForCausalLM.from_pretrained(config['model_name'], torch_dtype=torch.float16).to(device)
        self.model.config.sliding_window = 1024
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        add_all_special_tokens(self.tokenizer)

        self.questiongen_inference_params = config['questiongen_inference_params']
        self.answergen_inference_params = config['answergen_inference_params']

        self.verifier_model = AutoModelForSeq2SeqLM.from_pretrained(config['verifier_model_name'], torch_dtype=torch.float16).to(device)
        self.verifier_tokenizer = AutoTokenizer.from_pretrained(config['verifier_model_name'])
        yn_tokens = self.verifier_tokenizer.tokenize("yes no")
        self.yn_token_ids = self.verifier_tokenizer.convert_tokens_to_ids(yn_tokens)
        logger.info(f"Loaded {config['verifier_model_name']} verifier ({calculate_model_size_gb(self.verifier_model):.2f} GB)")
        logger.info(f"Initialized {self.agent_name} Agent ({calculate_model_size_gb(self.model):.2f} GB)")

        logger.info("-"*100)

    def __str__(self) -> str:
        return f"Agent: {self.agent_name}"

    def add_tokens(self, new_tokens: List):
        self.tokenizer.add_tokens(new_tokens)
        logger.info(f"Added new tokens to Agent: {', '.join(new_tokens)}")

    def remove_question_number(self, question):
        return re.sub(r'[0-9]+\.', '', question).strip()

    def stringify_questiongen_prompt(self, input_messages):
        prompt_string = ""
        for message in input_messages:
            if message['role'] == 'system':
                prompt_string += "[INST] " + message['content'] + "[/INST]"
            elif message['role'] == 'user':
                if message['content'] == "Describe the image":
                    prompt_string += message['content'] + " \n"
                else:
                    prompt_string += "Answer: " + message['content'] + " \n "
            elif message['role'] == 'assistant':
                prompt_string += "Question: " + message['content'] + " \n"
        assert input_messages[-1]['role'] != 'assistant'
        #prompt_string += 'Assistant: '
        return prompt_string
    
    def generate_multiple_nextquestions(self, state: List, num_questions: int = 1, include_last_agent_guess = False) -> str:
        init_sys_prompt = {
            'role': 'system', 
            'content': self.questiongen_prompts['init_sys_prompt']
        }
        final_sys_prompt = {
            'role': 'system', 
            'content': self.questiongen_prompts['final_sys_prompt'].replace('NUM_QUESTIONS', str(num_questions)).replace('VISUAL_QUERY', state.visual_query.question)
        }

        input_messages = [init_sys_prompt]
        input_messages += state.messages[1:] + [final_sys_prompt]
        prompt_string = self.stringify_questiongen_prompt(input_messages)
        input_ids = self.tokenizer(prompt_string, return_tensors='pt', truncation=True).input_ids.to(self.device)
        outputs = self.model.generate(input_ids, num_return_sequences=num_questions, **self.questiongen_inference_params)
        questions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pdb.set_trace()
        return questions

    def generate_supportingevidence_questions(self, state: List, num_questions: int = 1, possible_answer: str = None) -> str:
        init_sys_prompt = {
            'role': 'system', 
            'content': self.questiongen_prompts['init_sys_prompt']
        }
        final_sys_prompt = self.questiongen_prompts['final_sys_prompt'].replace('NUM_QUESTIONS', str(num_questions)).replace('VISUAL_QUERY', state.visual_query.question).replace('POSSIBLE_ANSWER', possible_answer)
        final_sys_prompt = {
            'role': 'system', 
            'content': final_sys_prompt
        }

        input_messages = [init_sys_prompt]
        for example in self.in_context_examples:
            input_messages += example
        input_messages += state.messages + [final_sys_prompt]

        # Transformer ChatGPT-style messages into Mistral-style messages
        new_input_messages = []
        for m in input_messages:
            if m['role'] == 'system':
                m['role'] = 'user'
                new_input_messages.append(m)
            else:
                if m['role'] == 'user':
                    if m['content'].startswith("Visual Query:"):
                        message_content = m['content'] + " \n"
                    else:
                        message_content = "Answer: " + m['content'] + " \n "
                elif m['role'] == 'assistant':
                    message_content = "Question: " + m['content'] + " \n "
                if new_input_messages[-1]['role'] == 'assistant':
                    new_input_messages[-1]['content'] += message_content
                else:
                    new_input_messages.append({
                        'role': 'assistant',
                        'content': message_content
                    })

        #prompt_string = self.stringify_questiongen_prompt(input_messages)
        input_ids_encoded = self.tokenizer.apply_chat_template(new_input_messages, return_tensors="pt").to(self.device)
        outputs = self.model.generate(input_ids_encoded, num_return_sequences=num_questions, **self.questiongen_inference_params)
        
        generated_ids = outputs[0, input_ids_encoded.shape[1]+1:]
        questions = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        questions = [x.strip() for x in questions.split('\n')]
        questions = [q for q in questions if len(q) > 0]
        questions = [self.remove_question_number(q) for q in questions]
        questions = [q for q in questions if q.endswith('?')]
        return questions


QGEN_CLASS_MAP = {
    'gpt': GPTQuestionGenerator,
    #'mistral': MistralQuestionGenerator
}