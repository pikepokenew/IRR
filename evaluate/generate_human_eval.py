import os
import time
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader



import sys
from human_eval.data import read_problems, write_jsonl

from src import system_template

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig # add
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model under evaluation: gpt4, chatgpt, huggingface_model_path', type=str, required=True)
parser.add_argument('--lora_path', help='', type=str, required=False, default=None)
parser.add_argument('--save_path', help='path where the model results to be saved', type=str, required=False, default='evaluate/results')
parser.add_argument('--num_samples', help='number of first num_samples to test from the dataset', type=int, required=False, default=-1)
parser.add_argument('--batch_size', help='batch_size', required=False, type=int, default=8)
parser.add_argument('--use_system_prompt', help='path to harmful questions (json) for evaluation, to be used with prompt templates for red-teaming', action="store_true")
parser.add_argument('--num_samples_per_task', help='num_samples_per_task', required=False, type=int, default=1)
parser.add_argument('--use_cot_prompt', help='CoT', action="store_true")
parser.add_argument('--use_chat_template', help='Chat Template', action="store_true")

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
# llama_tokenizer = AutoTokenizer.from_pretrained("/home/xxx/local_models/Llama-2-7b-chat-hf", padding_side="left", use_fast=False)
# tokenizer.chat_template = llama_tokenizer.chat_template
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
if args.lora_path != None:
    adapter_model_id  = args.lora_path
    print("Load Lora from:{}".format(adapter_model_id))
    model = PeftModel.from_pretrained(model, adapter_model_id)
##define chat completion function for GPT##

##define chat completion function for Claude##

##process data
# def clean_thoughts_(response):

#     if "(Internal thought:" in response:
#         if ')' in response:
#             ind =  response.index(')')+1
#         else:
#             ind = -1
#         nresponse = response[ind:].strip()
#         return nresponse

#     return response


# def get_context(file_name):
#     f = open(file_name, "r")
#     f = f.read()
#     return f

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
dataset = "/home/xxx/human-eval/data/HumanEval.jsonl.gz"
problems = read_problems(dataset)
# import pdb; pdb.set_trace()
# prompt_que, orig_que, topics, subtopics = process_data(dataset, num_samples)


##generate responses##
if not os.path.exists(save_path):
    os.makedirs(save_path)

#save file name
if args.lora_path != None:
    save_name = f'{save_path}/{dataset.split("/")[-1].replace(".jsonl.gz","")}/{model_name.split("/")[-1]}_{args.lora_path.split("/")[-1]}_sys_{args.use_system_prompt}_@{args.num_samples_per_task}_chat_{args.use_chat_template}.jsonl'
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
                # chat_template = data
                data_item = self.tokenizer(chat_template, padding=True, return_tensors="pt", add_special_tokens = False)
            else:
                data_item = self.tokenizer(data, padding=True, return_tensors="pt")

            data_item = {k: v[0] for k, v in data_item.items()}
            # data_item = {"task_id": task_id}
            data_item = [data_item] * num_samples_per_task
            # import pdb; pdb.set_trace()
            self.features.extend(data_item)
            # task_id = 
            self.task_id_list.extend([task_id] * num_samples_per_task)
            self.prompt_list.extend([data] * num_samples_per_task)
            count += 1
            if count == 1:
                decode_text = self.tokenizer.decode(data_item[0]["input_ids"].tolist(), skip_special_tokens = False)
                print(f"chat_template:{[decode_text]}")

                
        # self.features = self.features[:20]
        # self.task_id_list = self.task_id_list[:20]

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]

problems = read_problems("HumanEval.jsonl.gz")
# import pdb; pdb.set_trace()
num_samples_per_task = args.num_samples_per_task

if args.use_system_prompt == True:
    if "llama" in args.model.lower():
        system_prompt = system_template.llama_2_system_prompt
else:
    system_prompt = None

human_eval_dataset = HumanEvalDataset(problems, tokenizer, num_samples_per_task = num_samples_per_task, system_prompt=system_prompt, use_cot_prompt = args.use_cot_prompt, use_chat_template = args.use_chat_template, num_samples = num_samples)
data_collator = DataCollatorWithPadding(tokenizer, padding = "longest")
human_eval_dataloader = DataLoader(human_eval_dataset, batch_size = batch_size, collate_fn=data_collator)

outputs_list = []
response_text_list = []
generation_util = [
    "</s>",
    "<|im_end|>",
    "<|eot_id|>",
]
for batch in tqdm(human_eval_dataloader, total=len(human_eval_dataloader)):
    with torch.no_grad():
        # batch['input_ids'].to("cuda")
        # batch['attention_mask'].to("cuda")
        # import pdb; pdb.set_trace()
        for k, v in batch.items():
            batch[k] = v.to("cuda")
        if num_samples_per_task == 1:
            outputs = model.generate(
                **batch,
                max_new_tokens=512,
                stop_strings = generation_util,
                # temperature=0.2,
                # top_p=0.95,
                # do_sample=True,
                # temperature = 0,
                # do_sample=False,
                tokenizer=tokenizer,
                use_cache=True
            )
        else:
            outputs = model.generate(
                **batch,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                stop_strings = generation_util,
                tokenizer=tokenizer,
                use_cache=True
            ) 
        # import pdb; pdb.set_trace()
        seq_len = batch['input_ids'].shape[-1]
        # import pdb; pdb.set_trace()
        outputs = outputs[:,seq_len:]
        # import pdb; pdb.set_trace()
        outputs_list.extend(outputs.tolist())
outputs = []
samples = []

def filter_code_v1(completion: str) -> str:
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

def filter_code_v2(completion: str) -> str:
    # completion = completion.lstrip("\n")
    # return completion.split("\n\n")[0]
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

for idx, response_ids in enumerate(outputs_list):
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    if args.use_chat_template == True:
        response = filter_code_v2(response)
    else:
        response = filter_code_v1(response)
    samples.append(dict(task_id = human_eval_dataset.task_id_list[idx], prompt=human_eval_dataset.prompt_list[idx], completion=response,))


from typing import Iterable, Dict
import gzip

write_jsonl(save_name, samples)
print(f"\nCompleted, please check {save_name}")
