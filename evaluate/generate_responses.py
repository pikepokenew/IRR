import os
import time
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上上级目录的路径
parent_parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))

# 添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.system_template import llama_2_system_prompt

from transformers import AutoTokenizer
# import deepspeed

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Specify the model to be evaluated. Options include: gpt4, chatgpt, or a Hugging Face model path.', type=str, required=True)
parser.add_argument('--save_path', help='Directory path where the evaluation results will be saved. Default is "evaluate/results".', type=str, required=False, default='evaluate/results')
parser.add_argument('--save_name', help='Filename under which the model results will be saved. If not provided, a default name will be generated.', type=str, required=False, default=None)
parser.add_argument('--num_samples', help='Specify the number of samples to test from the dataset. Default is -1, which means all samples will be used.', type=int, required=False, default=-1)
parser.add_argument('--dataset', help='Path to the JSON file containing harmful questions for evaluation. This file will be used with prompt templates for red-teaming.', required=True, type=str)
parser.add_argument('--need_system_prompt', help='Indicate whether a system prompt is needed (1 for yes, 0 for no). Default is 0 (no).', required=False, type=int, default=0)
parser.add_argument('--max_new_tokens', help='Indicate whether a system prompt is needed (1 for yes, 0 for no). Default is 0 (no).', required=False, type=int, default=512)

args = parser.parse_args()

dataset = args.dataset
model_name = args.model
save_path = args.save_path
num_samples = args.num_samples

if args.need_system_prompt == 0:
    system_prompt = None
else:
    if "open-instruct-llama2-sharegpt-dpo-7b" in args.model or "tulu" in args.model.lower():
        system_prompt = None
    elif "llama" in args.model.lower() or "llama2" in args.model.lower():
        system_prompt = llama_2_system_prompt
    elif "mistral" in args.model.lower():
        system_prompt = None
    else:
        system_prompt = None
    
print(f"\n\nconfiguration")
print(f"*{'-'*10}*")

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

print(f"*{'-'*10}*\n\n")


##setting up model##

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=False)
    
tokenizer.pad_token = tokenizer.eos_token

llm = LLM(model_name)

def gen_prompt(que, system_prompt = None):
    chat = []
    if system_prompt != None and system_prompt != "" :
        chat.append({"role": "system", "content": system_prompt})
    chat.append({"role": "user", "content": que})
    # import pdb; pdb.set_trace()
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt = True)
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
                    prompt_que.append(gen_prompt(q, system_prompt = system_prompt))
                    topics.append(topic)
                    subtopics.append(subtopic)
    elif "salad" in dataset.lower():
        if "attack_enhanced_set" in dataset.lower():
            data_key = "prompt"
        elif "base_set" in dataset.lower():
            data_key = "prompt"
        prompt_que = [gen_prompt(q[data_key], system_prompt=system_prompt) for q in data]
        orig_que = data
        topics, subtopics = [], []
    elif "wild_jailbreak" in dataset.lower():
        # import pdb; pdb.set_trace()
        prompt_que = [gen_prompt(q["prompt"], system_prompt=system_prompt) for q in data]
        orig_que = data
        topics, subtopics = [], []
    elif type(data[0]) == dict and data[0].get("instruction", None) != None and data[0].get("input", None) != None:
        # prompt = ""
        prompt_que = [gen_prompt('''{}\n{}'''.format(q["instruction"], q['input']), system_prompt=system_prompt) for q in data]
        orig_que = data
        topics, subtopics = [], []
    else:
        prompt_que = [gen_prompt(q, system_prompt=system_prompt) for q in data]
        orig_que = data
        topics, subtopics = [], []

    if nsamples == -1:
        nsamples = len(prompt_que)

    return prompt_que[:nsamples], orig_que[:nsamples], topics[:nsamples], subtopics[:nsamples]


prompt_que, orig_que, topics, subtopics = process_data(dataset, num_samples)


##generate responses##
if not os.path.exists(save_path):
    os.makedirs(save_path)

#save file name
if args.save_name != None:
    save_name = f"{save_path}/{args.save_name}"

else:

    if not os.path.exists(f'{save_path}/{dataset.split("/")[-1].replace(".json","")}'):
        os.makedirs(f'{save_path}/{dataset.split("/")[-1].replace(".json","")}')

    save_name = f'{save_path}/{dataset.split("/")[-1].replace(".json","")}/{model_name.split("/")[-1]}_sys_{args.need_system_prompt}.json'
outputs = []
# system_message = ''

stop_words = ["</s>"]
stop_words.append(tokenizer.eos_token)
if "llama3" in model_name.lower() or "llama-3" in model_name.lower():
    stop_words.append("<|eot_id|>")

print("generating responses...\n")

outputs = []

temperature = 0.00
top_p = 1.0
max_tokens = args.max_new_tokens
prompts = prompt_que

print("----------------------------------------------------")
print("input:\n{}".format(prompts[0]))
print("----------------------------------------------------")

outputs = llm.generate(prompts, SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=1,
                stop=stop_words,
))
outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
outputs_list = [output.outputs[0].text for output in outputs]

# import pdb; pdb.set_trace()
response_text_list = []
outputs = []
for idx, response in enumerate(outputs_list):
    response_text_list.append(response)

    question = orig_que[idx]
    question2 = prompt_que[idx].replace('<s>','')
    
    response = response.replace(question2,"").strip()

    if 'zephyr' in model_name:
        response = response[response.index(question2[-10:]):][10:]

    if 'harmfulq' in dataset or 'cat' in dataset:
        response = [{'prompt':question, 'response':response, 'topic':topics[idx], 'subtopic': subtopics[idx]}]
    else:
        response = [{'prompt':question, 'response':response}]

    outputs += response

with open(f'{save_name}', 'w', encoding='utf-8') as f:
    json.dump(outputs, f, ensure_ascii=False, indent=4)



print(f"\nCompleted, pelase check {save_name}")
