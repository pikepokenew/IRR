import os
import time
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams


import sys
from human_eval.data import read_problems, write_jsonl

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src import system_template

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig # add
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model under evaluation: gpt4, chatgpt, huggingface_model_path', type=str, required=True)
parser.add_argument('--save_path', help='path where the model results to be saved', type=str, required=False, default='evaluate/results')
parser.add_argument('--num_samples', help='number of first num_samples to test from the dataset', type=int, required=False, default=-1)
parser.add_argument('--batch_size', help='batch_size', required=False, type=int, default=8)
parser.add_argument('--use_system_prompt', help='path to harmful questions (json) for evaluation, to be used with prompt templates for red-teaming', action="store_true")
parser.add_argument('--num_samples_per_task', help='num_samples_per_task', required=False, type=int, default=1)
parser.add_argument('--use_cot_prompt', help='CoT', action="store_true")
parser.add_argument('--use_chat_template', help='whether using chat template', action="store_true")

args = parser.parse_args()

batch_size = args.batch_size
model_name = args.model
save_path = args.save_path
num_samples = args.num_samples

print(f"\n\nconfiguration")
print(f"*{'-'*10}*")

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

print(f"*{'-'*10}*\n\n")


##setting up model##


tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model = LLM(model_name)


def gen_prompt(que):
    chat = [{"role": "user", "content": que}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    return prompt

def process_data(dataset, nsamples):
    f = open(dataset)

    data = json.load(f)

    if 'harmfulq' in dataset or 'cat' in dataset:
        topics = []
        subtopics = []
        prompt_que = []
        orig_que = []
        for topic in data.keys():
            for subtopic in data[topic].keys():
                for q in data[topic][subtopic]:
                    orig_que.append(q)
                    prompt_que.append(gen_prompt(q))
                    topics.append(topic)
                    subtopics.append(subtopic)

    else:
        prompt_que = [gen_prompt(q) for q in data]
        orig_que = data
        topics, subtopics = [], []

    if nsamples == -1:
        nsamples = len(prompt_que)

    return prompt_que[:nsamples], orig_que[:nsamples], topics[:nsamples], subtopics[:nsamples]
dataset = "/home/dwu/human-eval/data/HumanEval.jsonl.gz"
problems = read_problems(dataset)
# import pdb; pdb.set_trace()
# prompt_que, orig_que, topics, subtopics = process_data(dataset, num_samples)


##generate responses##
if not os.path.exists(save_path):
    os.makedirs(save_path)

#save file name
else:
    save_name = f'{save_path}/{dataset.split("/")[-1].replace(".jsonl.gz","")}/{model_name.split("/")[-1]}_sys_{args.use_system_prompt}_@{args.num_samples_per_task}_chat_{args.use_chat_template}.jsonl'
outputs = []
system_message = ''


print("generating responses...\n")


class HumanEvalDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, tokenizer, system_prompt=None, num_samples_per_task = 16, use_cot_prompt = False, use_chat_template = False, num_samples = -1):
        
        # if num_samples != -1:
        #     data_list = data_list[:num_samples]
        self.data_list = []
        for _, data in data_list.items():
            # task_id = data['task_id']

            # prompt = data['prompt']
            # data = data['prompt'].replace("    ", "\t")
            # data = data['prompt']
            self.data_list.append({"task_id": data['task_id'], "prompt": data["prompt"]})
        if num_samples != -1:
            self.data_list = self.data_list[:num_samples]   

        # self.data_list = data_list
        self.tokenizer = tokenizer
        self.features = []
        self.task_id_list = []
        self.prompt_list = []
        count = 0
        for data in self.data_list:
            
            task_id = data['task_id']
            data = data['prompt']
            # import pdb; pdb.set_trace()
            if use_cot_prompt:
                data = data + "\nLet's think step by step."
            # import pdb; pdb.set_trace()
            if use_chat_template == True:
                instruction = "Complete the following Python code without any tests or explanation"
                data = f'''{instruction}\n{data}'''
                messages = []
                if system_prompt != None and system_prompt != "" :
                    messages.append({"role": "system", "content": system_prompt})

                messages.append({"role": "user", "content": data})
                chat_template = tokenizer.apply_chat_template(messages, add_generation_prompt = True, tokenize = False)
            else:
                # data_item = self.tokenizer(data, padding=True, return_tensors="pt")
                pass    
            self.task_id_list.extend([task_id] * num_samples_per_task)
            self.prompt_list.extend([chat_template] * num_samples_per_task)
            count += 1
            if count == 1:
                print(f"chat_template:{[chat_template]}")

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]


# problems = read_problems(dataset)
# import pdb; pdb.set_trace()
num_samples_per_task = args.num_samples_per_task

if args.use_system_prompt == True:
    if "llama" in args.model.lower():
        system_prompt = system_template.llama_2_system_prompt
else:
    system_prompt = None

human_eval_dataset = HumanEvalDataset(problems, tokenizer, num_samples_per_task = num_samples_per_task, system_prompt=system_prompt, use_cot_prompt = args.use_cot_prompt, use_chat_template = args.use_chat_template, num_samples = num_samples)

outputs_list = []
response_text_list = []
generation_util = [
    "</s>",
    "<|im_end|>",
    "<|eot_id|>",
]

prompts = human_eval_dataset.prompt_list
if num_samples_per_task == 1:
    temperature = 0.0
else:
    temperature = 0.2
top_p = 1.0
max_tokens = 512
outputs = model.generate(prompts, SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=1,
                stop=generation_util,
))
outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
outputs_list = [output.outputs[0].text for output in outputs]

outputs = []
samples = []

def filter_code_v1(completion: str) -> str:
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

def filter_code_v2(completion: str) -> str:

    completion = completion.lstrip("\n")
    import re
    # import pdb; pdb.set_trace()

    code_blocks = re.findall(r'```(.*?)```', completion, re.DOTALL)
    if len(code_blocks) == 0:
        code_blocks = completion.split("\n")
        mark_idx = 0
        for idx, line in enumerate(code_blocks):
            if "def" in line:
                mark_idx = idx
        new_code = "\n".join(code_blocks[mark_idx + 1:])
    else:
        code_blocks = code_blocks[0]
        code_blocks = code_blocks.split("\n")
        mark_idx = 0
        for idx, line in enumerate(code_blocks):
            if "def" in line:
                mark_idx = idx
        new_code = "\n".join(code_blocks[mark_idx + 1:])
    return new_code

for idx, response in enumerate(outputs_list):
    # import pdb; pdb.set_trace()
    # response = tokenizer.decode(response_ids, skip_special_tokens=True)

    # response = response.replace("\t", "    ")
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    if args.use_chat_template == True:
        # response = filter_code_v1(response)
        response = filter_code_v2(response)
    else:
        response = filter_code_v1(response)
    # import pdb; pdb.set_trace()
    samples.append(dict(task_id = human_eval_dataset.task_id_list[idx], prompt=human_eval_dataset.prompt_list[idx], completion=response,))


from typing import Iterable, Dict
import gzip

write_jsonl(save_name, samples)
print(f"\nCompleted, please check {save_name}")
