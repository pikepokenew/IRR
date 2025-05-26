import torch
import re
import os
import argparse
import random
from tqdm import tqdm
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig # add
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteriaList,
    DataCollatorWithPadding,
)
from utils import (
    SpecificStringStoppingCriteria,
    extract_predicted_answer,
    extract_ground_truth
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from collections import Counter
import json
from vllm import LLM, SamplingParams
  

class GSM8KDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, tokenizer, template = None ,system_prompt=None, num_samples = -1):
        self.data_list = data_list
        self.tokenizer = tokenizer
        # self.features = []
        self.prompts = []
        for idx, data in enumerate(self.data_list):
            # chat_template = data
            question = data["question"]
            messages = []
            if system_prompt != None and system_prompt != "" :
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": question})
            chat_msg = tokenizer.apply_chat_template(messages, tokenize=False)
            if idx == 0:
                pass
                # decode_text = self.tokenizer.decode(data_item["input_ids"].tolist(), skip_special_tokens = False)
                print(f"chat_template:{chat_msg}")
                # import pdb; pdb.set_trace()
            # self.features.append(data_item)
            self.prompts.append(chat_msg)

        if num_samples != -1:
            self.prompts = self.prompts[ : num_samples ]
        # self.features = self.features[:20]
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index]

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1', 
                        help='Path or name of the model to be used for inference, such as a Hugging Face model identifier or a local model path.')
    parser.add_argument('--use_majority_vote', action='store_true', 
                        help='Use majority voting among multiple model answers to determine the final answer. If specified, n_votes should also be provided.')
    parser.add_argument('--lora_path', help='Optional path to the LoRA model for parameter-efficient tuning. If not provided, standard model weights will be used.', 
                        type=str, required=False, default=None)
    parser.add_argument("--temp", type=float, default=0, 
                        help='Temperature parameter for sampling. Higher values result in more random outputs; default is 0 (deterministic).')
    parser.add_argument('--n_votes', type=int, default=1, 
                        help='Number of votes to collect when using majority voting. This parameter is relevant only if --use_majority_vote is set.')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size for processing inputs. A larger batch size may improve throughput but requires more memory.')
    parser.add_argument('--system_prompt', type=int, default=0, 
                        help='Flag indicating whether to use a system prompt (1 for yes, 0 for no). Default is 0, meaning no system prompt will be used.')
    parser.add_argument('--num_samples', type=int, default=-1, 
                        help='Number of samples to be processed from the dataset. Default is -1, meaning all samples will be used.')
    parser.add_argument("--use_cot_prompt", action="store_true", 
                        help='Use a prompt template that encourages step-by-step reasoning (chain of thought). If specified, this will alter the prompt format.')

    args = parser.parse_args()
    print(f"\n\nconfiguration")
    print(f"*{'-'*10}*")

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print(f"*{'-'*10}*\n\n")
    batch_size = args.batch_size
    if args.system_prompt == 0 or "tulu" in args.model.lower():
        system_prompt = None
    else:
        # system_prompt = '''You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.'''
        system_prompt = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
        ),
        system_prompt = "".join(system_prompt)
        # system_prompt = ''''''
    device = "cuda"
    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    print('Loading model and tokenizer...')

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")

    tokenizer.pad_token = tokenizer.eos_token
    if "tulu" in args.model:
        tokenizer.chat_template = '''{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '<|user|>\n' + message['content'].strip() + '\n<|assistant|>'}}
    {% elif message['role'] == 'assistant' %}
        {{ message['content'] + eos_token }}
    {% endif %}
{% endfor %}
'''
    model = LLM(args.model) 

    print('\nLoading dataset...')
    dataset = load_dataset("openai/gsm8k", "main")["test"]
    datasize = len(dataset)
    print('gsm8k test size:', datasize) 
    generation_util = [
        "Q:",
        "</s>",
        "<|im_end|>",
        "<|eot_id|>",
    ]

    temperature = 0.0
    top_p = 1.0
    max_tokens = 512

    results = []
    if args.use_cot_prompt:
        template = "Q: {question}\nA: Let's think step by step."
    else:
        template = 'Q: {question}\nA:'
    gsm8k_dataset = GSM8KDataset(dataset, tokenizer = tokenizer, template = template, system_prompt = system_prompt, num_samples = args.num_samples)

    all_answers = []
    if args.use_majority_vote:
        for _ in range(args.n_votes):
            model_answers = []
            prompts = gsm8k_dataset.prompts
            outputs = model.generate(prompts, SamplingParams(
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            n=1,
                            stop=generation_util,
            ))
            outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
            outputs_list = [output.outputs[0].text for output in outputs]
            for text in outputs_list:
                text = text.split("A:")[-1].strip()
                model_answer = extract_predicted_answer(text)
                model_answers.append({"text": text, "numeric": model_answer})
            
            all_answers.append(model_answers)
    else:
        model_answers = []
        prompts = gsm8k_dataset.prompts
        outputs = model.generate(prompts, SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        n=1,
                        stop=generation_util,
        ))
        outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
        outputs_list = [output.outputs[0].text for output in outputs]

        for text in outputs_list:
            text = text.split("A:")[-1].strip()
            model_answer = extract_predicted_answer(text)
            model_answers.append({"text": text, "numeric": model_answer})
        all_answers.append(model_answers)
    results = []
    for ii in range(len(all_answers[0])):
        model_answers = [{"text": all_answers[idx][ii]['text'], "numeric": all_answers[idx][ii]['numeric']} for idx in range(args.n_votes)]

        numeric_answers = [ma['numeric'] for ma in model_answers]
        filtered_answers = [num for num in numeric_answers if num is not None]

        majority_answer = Counter(filtered_answers).most_common(1)[0][0] if filtered_answers else None

        ground_truth_answer = extract_predicted_answer(gsm8k_dataset.data_list[ii]['answer'])
        correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False

        results.append({
            'question': gsm8k_dataset.data_list[ii]['question'],
            'gold_answer_text': gsm8k_dataset.data_list[ii]['answer'],
            'model_answers_text': [ma['text'] for ma in model_answers],
            'extracted_model_answers': numeric_answers,
            'extracted_gold_answer': ground_truth_answer,
            'majority_answer': majority_answer,
            'correct': correct
        })

    cnt = 0
    for result in results:
        # print(result['correct'])
        if result['correct']:
            cnt += 1
    total = len(results)
    print(f"Accuracy: {cnt} / {total} = {cnt / total :.4f}")
    
    # import pdb; pdb.set_trace()
    accuracy = cnt / total
    # results.append({'accuracy': cnt / total})

    os.makedirs('results/gsm8k/zero_shot', exist_ok=True)
    if args.lora_path == None:
        model_name = args.model.split('/')[-1]
    else:
        model_name = args.model.split('/')[-1] + "_" + args.lora_path.split("/")[-1]

    result_file = f"results/gsm8k/zero_shot/{model_name}"
    if args.use_cot_prompt:
        result_file += "_cot"
    if args.use_majority_vote:
        result_file += f"_maj1@{args.n_votes}_temp{args.temp}"
    if args.system_prompt != 0:
        result_file += "_with_sys_prompt"
    # result_file += "_results.json"
    result_fold = result_file
    os.makedirs(result_fold, exist_ok=True)
    detail_file = result_fold + "/gsm8k_predict_detail.json"
    result_file = result_fold + "/gsm8k_results.json"

    with open(detail_file, 'w') as f:
        json.dump(results, f, indent=4)

    result_dict = {
        "accuracy": accuracy,
    }
    with open(result_file, 'w') as f:
        json.dump(result_dict, f, indent=4)

    print(f"Pred Detail saved to {detail_file}")
    print(f"Pred Results saved to {result_file}")
                

if __name__ == '__main__':
    main()


