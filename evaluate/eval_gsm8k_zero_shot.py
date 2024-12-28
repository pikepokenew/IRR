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
  

class GSM8KDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, tokenizer, template = None ,system_prompt=None, num_samples = -1):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.features = []
        for idx, data in enumerate(self.data_list):
            # chat_template = data
            question = data["question"]

            messages = []
            if system_prompt != None and system_prompt != "" :
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": question})
            chat = tokenizer.apply_chat_template(messages, tokenize=False)
            data_item = self.tokenizer(chat, padding=True, return_tensors="pt", add_special_tokens = False)
            data_item = {k: v[0] for k, v in data_item.items()}
            # data_item['answer'] = answer
            # data_item["input_text"] = question
            if idx == 0:
                pass
                decode_text = self.tokenizer.decode(data_item["input_ids"].tolist(), skip_special_tokens = False)
                print(f"chat_template:{decode_text}")
                # import pdb; pdb.set_trace()
            self.features.append(data_item)

        if num_samples != -1:
            self.features = self.features[ : num_samples ]
        # self.features = self.features[:20]
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--use_majority_vote', action='store_true')
    parser.add_argument('--lora_path', help='', type=str, required=False, default=None)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument('--n_votes', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--system_prompt', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument("--use_cot_prompt", action="store_true")
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    except:
        tokenizer = AutoTokenizer.from_pretrained("/home/dwu/local_models/Llama-2-7b-chat-hf", padding_side="left")
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
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.float16) 
    # import pdb; pdb.set_trace()
    if args.lora_path != None:
        adapter_model_id  = args.lora_path
        print("Load Lora from:{}".format(adapter_model_id))
        model = PeftModel.from_pretrained(model, adapter_model_id)
    print('\nLoading dataset...')
    dataset = load_dataset("openai/gsm8k", "main")["test"]
    datasize = len(dataset)
    print('gsm8k test size:', datasize) 
    model.eval()
    # Define a stopping condition for generation
    generation_util = [
        "Q:",
        "</s>",
        "<|im_end|>",
        "<|eot_id|>",
    ]

    results = []
    if args.use_cot_prompt:
        template = "Q: {question}\nA: Let's think step by step."
        
        # input_text = "Q: {question}\nA: Let's think step by step.".format(question=example['question'])
    else:
        # input_text = 'Q: ' + example['question'] + '\nA:'
        template = 'Q: {question}\nA:'
    # if args.num_samples != -
    gsm8k_dataset = GSM8KDataset(dataset, tokenizer = tokenizer, template = template, system_prompt = system_prompt, num_samples = args.num_samples)
    
    data_collator = DataCollatorWithPadding(tokenizer, padding = "longest")
    gsm8k_dataloader = DataLoader(gsm8k_dataset, batch_size = batch_size, collate_fn=data_collator)
    all_answers = []
    if args.use_majority_vote:
        for _ in range(args.n_votes):
            model_answers = []
            for batch in tqdm(gsm8k_dataloader, total=len(gsm8k_dataloader)):
                seq_len = batch["input_ids"].shape[-1]

                batch["input_ids"] = batch["input_ids"].to(device)
                batch["attention_mask"] = batch["attention_mask"].to(device)

                # import pdb; pdb.set_trace()
                # stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, seq_len)
                # stopping_criteria_list = StoppingCriteriaList([stop_criteria])
                with torch.no_grad():
                    outputs = model.generate(**batch, temperature=args.temp, max_new_tokens=512, do_sample=True, pad_token_id=tokenizer.eos_token_id, stop_strings = generation_util, tokenizer=tokenizer)
                outputs = outputs[:, seq_len:]
                output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for text in output_text:
                    text = text.split("A:")[-1].strip()
                    model_answer = extract_predicted_answer(text)
                    model_answers.append({"text": text, "numeric": model_answer})
            
            all_answers.append(model_answers)
                # model_answers.append({'text': output_text, 'numeric': model_answer})
                # all_answers.append({"text": text, "numeric": model_answer})
    else:
        model_answers = []
        for batch in tqdm(gsm8k_dataloader, total=len(gsm8k_dataloader)):
            seq_len = batch["input_ids"].shape[-1]
            # import pdb; pdb.set_trace()
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            # stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, seq_len)
            # stopping_criteria_list = StoppingCriteriaList([stop_criteria])
            with torch.no_grad():
                outputs = model.generate(**batch, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, stop_strings = generation_util, tokenizer=tokenizer, use_cache=True)
            # import pdb; pdb.set_trace()
            outputs = outputs[:, seq_len:]
            # import pdb; pdb.set_trace()
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for text in output_text:
                text = text.split("A:")[-1].strip()
                model_answer = extract_predicted_answer(text)
                model_answers.append({"text": text, "numeric": model_answer})
        all_answers.append(model_answers)
    results = []
    for ii in range(len(all_answers[0])):
        # import pdb; pdb.set_trace()
        model_answers = [{"text": all_answers[idx][ii]['text'], "numeric": all_answers[idx][ii]['numeric']} for idx in range(args.n_votes)]
        # import pdb; pdb.set_trace()
        numeric_answers = [ma['numeric'] for ma in model_answers]
        filtered_answers = [num for num in numeric_answers if num is not None]
        # import pdb; pdb.set_trace()
        majority_answer = Counter(filtered_answers).most_common(1)[0][0] if filtered_answers else None
        # import pdb; pdb.set_trace()
        ground_truth_answer = extract_predicted_answer(gsm8k_dataset.data_list[ii]['answer'])
        correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False
        # print(correct)
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


