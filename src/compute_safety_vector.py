import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='The model to evaluate, which can be a name from Hugging Face model hub (e.g., "gpt4", "chatgpt") or a local model path.', type=str, required=True)
    parser.add_argument('--base_model', help='The base model for comparison, also a name from Hugging Face model hub or a local path.', type=str, required=True)
    parser.add_argument('--save_path', help='The directory path where the model results will be saved. Default is "evaluate/results".', type=str, required=False, default='evaluate/results')
    parser.add_argument('--save_name', help='The filename to save the results. If not provided, a default name will be generated based on the model names.', type=str, required=False, default=None)
    parser.add_argument('--target', help='Specify the mode for evaluation: "all" for full parameter comparison or "lora" to target LoRA-specific parameters (e.g., "q_proj", "v_proj").', type=str, required=True, default="all")

    args = parser.parse_args()
    model_name = args.model 
    
    model = AutoModelForCausalLM.from_pretrained(model_name)

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    
    if args.target == "all":
        target_module = ["all"]
    elif args.target == "lora":
        target_module = ["q_proj", "v_proj"]
    print("target module: {}".format(target_module))
    task_vector = {}
    for name, param in model.named_parameters():
        if target_module == ["all"]:
            task_vector[name] = param.data
        else:
            for module in target_module:
                if module in name:
                    task_vector[name] = param.data

        # pass
    for name, param in base_model.named_parameters():
        if target_module == ["all"]:
            task_vector[name] = task_vector[name] - param.data
        else:
            for module in target_module:
                if module in name:
                    task_vector[name] = task_vector[name] - param.data
    
    # task_vector: {"model.layers.0.self_attn.q_proj.weight": [tensor], ...}
    # import pdb; pdb.set_trace()
    task_model_tag = model_name.split("/")[-1]
    base_model_tag = args.base_model.split("/")[-1]
    if args.save_name == None:
        folder_path = "task_vector"
        if not os.path.exists(folder_path):
            # 如果不存在，创建文件夹
            os.makedirs(folder_path)
        save_name = "task_vector/{}_from_{}_{}_task_vector.pth".format(task_model_tag, base_model_tag, "_".join(target_module))
    else:
        save_name = args.save_name
    print("task vector will be saved at: {}".format(save_name))
    torch.save(task_vector, save_name)
if __name__ == "__main__":
    main()
