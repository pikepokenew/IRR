import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm
import argparse
import os
from system_template import llama_2_system_prompt

# 自定义 Dataset 类，用于加载json数据
class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, tokenizer, system_prompt=None, num_samples = None):
        with open(data_dir, 'r') as f:
            data = json.load(f)
        self.data_list = data
        if num_samples != None and num_samples > 0 and num_samples <= len(self.data_list):
            self.data_list = data[:num_samples]

        # self.data_list = data[:1000]
        self.tokenizer = tokenizer
        self.features = []
        count_over_max_len = 0
        for ii, data in enumerate(self.data_list):
            # import pdb; pdb.set_trace()
            if data.get("instruction", None) != None and data.get("input", None) != None and data.get("output", None) != None:
                instruction = data['instruction']
                input = data['input']
                output = data['output']
            elif data.get("prompt", None) != None and data.get("response", None) != None:
                if data['prompt'].get("instruction", None) != None and data['prompt'].get("input", None) != None:
                    instruction = "{}\n{}\n".format(data['prompt']["instruction"], data['prompt']["input"])
                else:
                    instruction = "{}".format(data['prompt'])
                input = ""
                output = data['response']
            messages = []
            if system_prompt != None and system_prompt != "" :
                messages = [{"role": "system", "content": system_prompt}]
                pass
            if input == "":
                content = instruction
                messages.append({"role": "user", "content": content})
            else:
                content = "{}\n{}".format(instruction, input)
                messages.append({"role": "user", "content": content})
            chat_template = tokenizer.apply_chat_template(messages, add_generation_prompt = True, tokenize = False)

            input_ = self.tokenizer(chat_template, padding=True, return_tensors="pt", add_special_tokens = False)
            output_ = self.tokenizer(output, padding=True, return_tensors="pt", add_special_tokens = False)
            output_['input_ids'] = torch.cat((output_['input_ids'], torch.tensor([[self.tokenizer.eos_token_id]])), dim = -1)
            output_['attention_mask'] = torch.cat((output_['attention_mask'], torch.tensor([[1]])), dim = -1)
            
            # import pdb; pdb.set_trace()
            max_token_len = 1024

            # import pdb; pdb.set_trace()
            data_item = {k: torch.cat((v[0], output_[k][0]), dim=-1) for k, v in input_.items()}

            data_item['label'] = torch.cat((torch.full_like(input_['input_ids'][0], -100), output_['input_ids'][0]), dim=-1)

            if data_item["input_ids"].shape[-1] > max_token_len:
                data_item = {k: v[:max_token_len] for k, v in data_item.items()}
                count_over_max_len += 1  

            data_item = {k: v.long() for k, v in data_item.items()}
            if ii == 0:
                print(f"chat_template: {[chat_template]}")
                print(f"input text:{[self.tokenizer.decode(input_.input_ids.tolist()[0], add_special_tokens = False)]}\ntarget text:{[self.tokenizer.decode(output_.input_ids.tolist()[0], add_special_tokens = False)]}")
                # print(f)
                for k, v in data_item.items():
                    print(f"{k}: {v}")

            self.features.append(data_item)
        print(f"Get dataset size:{len(self.features)}\tTruncation size:{count_over_max_len}")

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]

# 加载 json 数据
def load_json_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 定义 Fisher 矩阵计算的函数
def compute_fisher_matrix(model, dataloader, tokenizer, device, targets):
    fisher_matrix = {}
    model.eval()  # 模型切换到评估模式，确保不会更新参数

    # 冻结模型的参数
    for name, param in model.named_parameters():
        found_flag = False
        if targets == ["all"]:
            found_flag = True
        else:
            for tgt in targets:
                if tgt in name: 
                    found_flag = True
                    break
        if found_flag == False:
            param.requires_grad = False

    for batch in tqdm(dataloader, total=len(dataloader)):
        
        # inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        for k, v in batch.items():
            batch[k] = v.to(device)
        # 前向传递
        outputs = model(**batch)
        loss = outputs.loss
        
        # 计算梯度
        loss.backward()
        
        # 计算每个参数的梯度平方
        for name, param in model.named_parameters():
            if param.grad is not None:
                if targets == ["all"]:
                    if name not in fisher_matrix:
                        fisher_matrix[name] = param.grad.pow(2).detach().cpu().clone()
                    else:
                        fisher_matrix[name] += param.grad.pow(2).detach().cpu().clone()
                else:
                    for tgt in targets:
                        if tgt in name:
                            if name not in fisher_matrix:
                                fisher_matrix[name] = param.grad.pow(2).detach().cpu().clone()
                            else:
                                fisher_matrix[name] += param.grad.pow(2).detach().cpu().clone()
        model.zero_grad()
    # 取平均值
    for name in fisher_matrix:
        fisher_matrix[name] /= len(dataloader)

    return fisher_matrix

# 主函数


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='The model to evaluate; can be from Hugging Face Model Hub (e.g., "gpt4", "chatgpt") or a local model path.', type=str, required=True)
    parser.add_argument('--save_path', help='Directory path where the model evaluation results will be saved. Defaults to "evaluate/results".', type=str, required=False, default='evaluate/results')
    parser.add_argument('--save_name', help='Filename for saving the results. If not provided, a default name will be generated based on the dataset and model names.', type=str, required=False, default=None)
    parser.add_argument('--num_samples', help='Number of samples to draw from the dataset for evaluation. Defaults to -1, which means all samples will be used.', type=int, required=False, default=-1)
    parser.add_argument('--dataset', help='Path to the JSON file containing data for evaluation. The questions should be appropriate for red-teaming tasks.', required=True, type=str)
    parser.add_argument('--batch_size', help='The number of samples to process in each batch during evaluation. Default is 8.', required=False, type=int, default=8)
    parser.add_argument('--need_system_prompt', help='Whether to include a system prompt for the model (1 for yes, 0 for no). Default is 0 (no system prompt).', required=False, type=int, default=0)
    parser.add_argument('--target', help='Specify the evaluation mode; options include "lora" for LoRA-specific parameters, or "all" for all parameters. Default is "lora".', type=str, required=False, default="lora")

    args = parser.parse_args()

    # 加载预训练模型和tokenizer
    if args.need_system_prompt == 1:
        system_prompt = llama_2_system_prompt
    elif args.need_system_prompt == 0:
        system_prompt = None
    model_name = args.model  # 选择合适的模型名称
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if args.target == "lora":
        targets = ["q_proj", "v_proj"]
    else:
        targets = ["all"]
    batch_size = args.batch_size
    num_samples = args.num_samples
    # 加载数据
    json_file = args.dataset 

    folder_path = "FIMs"

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 如果不存在，创建文件夹
        os.makedirs(folder_path)
    save_name = "{}/".format(folder_path) + json_file.split("/")[-1].split(".")[0] + "_{}".format(model_name.split("/")[-1]) + "_{}".format("_".join(targets)) + "_fisher_matrix.pth"
    print("save name: {}".format(save_name))

    # 创建 DataLoader
    dataset = CustomDataset(json_file, tokenizer=tokenizer, system_prompt=system_prompt, num_samples = num_samples)

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding="longest")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)  # 设置合适的batch_size

    # 计算 Fisher 矩阵
    fisher_matrix = compute_fisher_matrix(model, dataloader, tokenizer, device, targets)
    # fisher_matrix = {"model.layers.0.self_attn.q_proj.weight": [tensor], ...}
    # 输出 Fisher 矩阵保存结果
    print("fisher information matrix will be saved at: {}".format(save_name))
    torch.save(fisher_matrix, save_name)
if __name__ == "__main__":
    main()
