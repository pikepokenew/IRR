import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer
import json

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_tokenizer(model):
    if "llama" in model.lower():
        # tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def pad_or_truncate_tensor(sequence, max_len, pad_token_id):
    """
    Helper function to truncate or pad the tensor sequence.
    
    :param sequence: torch tensor of input ids or attention mask
    :param max_len: the target sequence length (seqlen)
    :param pad_token_id: the id for padding
    :return: processed torch tensor
    """
    current_len = sequence.size(0)
    
    # If sequence is longer than max_len, truncate it
    if current_len > max_len:
        return sequence[:max_len]
    
    # If sequence is shorter than max_len, pad it on the left
    padding_len = max_len - current_len
    padding = torch.full((padding_len,), pad_token_id, dtype=sequence.dtype)
    return torch.cat((padding, sequence), dim=0)


def get_dataset(path, nsamples, seed, seqlen, model, tokenizer, need_system_prompt = 0):
    if need_system_prompt == 1:
        system_prompt = '''You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.'''
    else:
        system_prompt = None
    tokenizer.pad_token_id = tokenizer.eos_token_id
    with open(path, "r") as f:
        dataset = json.load(f)
    if nsamples != -1:
        dataset = dataset[:nsamples]
    trainloader = []
    for data in dataset:
        chat_message = []
        if system_prompt != None:
            chat_message.append({"role": "system", "content": system_prompt})
        if data.get("instruction", None) != None and data.get("input", None) != None and data.get("output", None) != None:
            instruction = data['instruction']
            input = data['input']
            output = data['output']
        elif data.get("prompt", None) != None and data.get("response", None) != None:
            instruction = "{}\n{}\n".format(data['prompt']["instruction"], data['prompt']["input"])
            input = ""
            output = data['response']

        if input == "":
            content = instruction
            chat_message.append({"role": "user", "content": content})
        else:
            content = "{}\n{}".format(instruction, input)
            chat_message.append({"role": "user", "content": content})

        chat_template = tokenizer.apply_chat_template(chat_message, add_generation_prompt = True, tokenize = False)

        input_ = tokenizer(chat_template, padding=True, return_tensors="pt", add_special_tokens = False)
        output_ = tokenizer(output, padding=True, return_tensors="pt", add_special_tokens = False)
        output_['input_ids'] = torch.cat((output_['input_ids'], torch.tensor([[tokenizer.eos_token_id]])), dim = -1)
        output_['attention_mask'] = torch.cat((output_['attention_mask'], torch.tensor([[1]])), dim = -1)
        data_item = {k: torch.cat((v[0], output_[k][0]), dim=-1) for k, v in input_.items()}
        data_item['label'] = torch.cat((torch.full_like(input_['input_ids'][0], -100), output_['input_ids'][0]), dim=-1)
        
        
        data_item['input_ids'] = pad_or_truncate_tensor(data_item['input_ids'], max_len=seqlen, pad_token_id=tokenizer.pad_token_id)
        data_item['attention_mask'] = pad_or_truncate_tensor(data_item['attention_mask'], max_len=seqlen, pad_token_id=0)
        data_item['label'] = pad_or_truncate_tensor(data_item['label'], max_len=seqlen, pad_token_id=-100)
        # import pdb; pdb.set_trace()
        data_item = {k: v.int() for k, v in data_item.items()}
        trainloader.append((data_item['input_ids'], data_item['label'], data_item['attention_mask']))
        # instruction =  if data.get("instruction", None) != None
        # if data.get("instruction", None) != None and data[0].get("input", None) != None 
    testloader = None
    return trainloader, testloader
    


def get_c4(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model='', path = None, need_system_prompt = 1):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer)
    
    # if "gsm8k" in name:
    #     return get_gsm8k(nsamples, seed, seqlen, model, tokenizer)
    return get_dataset(name, nsamples, seed, seqlen, model, tokenizer, need_system_prompt)
