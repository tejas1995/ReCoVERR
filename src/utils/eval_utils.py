import json
from utils.openai_utils import openai_caller
from pathlib import Path
from typing import List, Any, Union, Tuple
from collections import Counter
from modules.lave.lave import LaveBase

#binary_prompts = json.load(open('/net/nfs.cirrascale/mosaic/tejass/data/lave_eval/binary_prompts.json'))
binary_prompts = json.load(open('/home/tejas/data/lave_eval/binary_prompts.json'))
#nonbinary_prompts = json.load(open('/net/nfs.cirrascale/mosaic/tejass/data/lave_eval/nonbinary_prompts.json'))
nonbinary_prompts = json.load(open('/home/tejas/data/lave_eval/nonbinary_prompts.json'))

def get_lave_score(question, reference_answers, candidate_answer):
    unique_ref_answers = list(set(reference_answers))
    unique_ref_answers.sort(key=lambda x: reference_answers.count(x), reverse=True)
    top_answer = unique_ref_answers[0]
    if top_answer in ['yes', 'no']:
        incontext_prompts = binary_prompts
    else:
        incontext_prompts = nonbinary_prompts

    input_messages = incontext_prompts + \
        [{'role': 'user', "content": f"Question: {question} \n Reference answers: {', '.join(reference_answers)} \n Candidate answer: {candidate_answer}"}]

    gpt_response = openai_caller(
                input_messages,        
                model='chatgpt-16k', 
                max_new_tokens=150, 
                temperature=0.0,
                num_completions=1
            )
    #import pdb; pdb.set_trace()
    try:
        rating = int(gpt_response.split('=')[-1])
        assert rating in [1, 2, 3]
    except:
        rating = 1
    score = (rating-1)/2
    reasoning = gpt_response
    return reasoning, score

class LaveChatGPT(LaveBase):
    
    def __init__(
        self,
        num_shots: int = 8,
        rationalize: bool = True,
        filter_refs: bool = True,
        use_caption: bool = False,
        demos_file: Union[str, Path] = '/home/tejas/data/lave_eval/nonbinary_prompts.json',
        binary_demos_file: Union[str, Path] = '/home/tejas/data/lave_eval/binary_prompts.json',
        debug: bool = False
    ) -> None:
        super().__init__(
            num_shots=num_shots,
            rationalize=rationalize,
            filter_refs=filter_refs,
            use_caption=use_caption,
            demos_file=demos_file,
            binary_demos_file=binary_demos_file,
            debug=debug
        )
        self.input_template = "Question: '{question}' \n Reference answers: {references} \n Candidate answer: '{prediction}'"
        self.output_template = "Output: {output}"

    def build_prompt(self, prediction: str, references: List[str], question: str, caption: str = None) -> str:
        prompt_messages = [{'role': 'system', 'content': self.task_definition}]
        demos = self.select_demos(question, references)
        for demo in demos:
            kwargs = {
                'question': demo['question'],
                'references': self.format_references(demo['references'], filter=False),
                'prediction': demo['prediction'],
                'output': f"{demo['explanation']} So rating={demo['output']}" if self.rationalize else demo['output']
            }
            prompt_messages.append({
                'role': 'user',
                'content': self.input_template.format(**kwargs)
            })
            prompt_messages.append({
                'role': 'assistant',
                'content': self.output_template.format(**kwargs)
            })

        kwargs = {
            'question': question,
            'references': self.format_references(references, filter=self.filter_refs),
            'prediction': prediction,
            'output': ''
        }
        prompt_messages.append({
            'role': 'user',
            'content': self.input_template.format(**kwargs)
        })
        return prompt_messages        


    def compute(self, prediction: str, references: List[str], question: str, caption: str = None) -> Tuple[str, float]:
        prompt_messages = self.build_prompt(prediction, references, question, caption)
        gpt_response = openai_caller(
            prompt_messages, 
            model='chatgpt-16k', 
            max_new_tokens=150, 
            temperature=0.0,
            num_completions=1
        )
        #import pdb; pdb.set_trace()
        try:
            rating = int(gpt_response.split('=')[-1])
            assert rating in [1, 2, 3]
        except:
            rating = 1
        score = (rating-1)/2
        reasoning = gpt_response
        return reasoning, score

lave_scorer = LaveChatGPT()