import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm
import argparse
import deepspeed
from deepspeed.utils import (
    safe_get_full_grad
    )

import copy

import numpy
import torch
from transformers import AutoModelForCausalLM

from dataclasses import dataclass, field


@dataclass
class SafeLoRAConfig:
    """
    This is the configuration class to store the configuration of a safeLoRA.
    """

    base_model_path: str = field(
        default=None,
        metadata={"help": "The path of the base model for obtaining the aligned matrix"},
    )

    unaligned_model_path: str = field(
        default=None,
        metadata={"help": "The path of the aligned model for obtaining the aligned matrix"},
    )


    select_layers_type: str = field(
        default="number",
        metadata={"help": "How to select projection layers? options: [threshold, number]"},
    )

    threshold: float = field(
        default=0.5,
        metadata={"help": "The threshold of cosine similarity."},
    )

    num_proj_layers: int = field(
        default=10,
        metadata={"help": "The number of projected layers."},
    )

    devices: str = field(
        default="cuda",
        metadata = {"help": "Devices are used in SafeLoRA. (gpu or cpu)"}

    )

    target: str = field(
        default="full",
    )
    

    # def __post_init__(self):
    #     if self.base_model_path is None:
    #         raise ValueError("base_model_path cannot be None.")
    #     if self.aligned_model_path is None:
    #         raise ValueError("aligned_model_path cannot be None.")

class SafeLoRA:
    def __init__(self, ft_model:torch.nn.Module, config):
        """
        Please use safelora.model to get the projected model.

        How to use SafeLoRA:
        path = './LLM_Models/llama-2-7b-chat-fp16/' # load your base model of the peft model
        model = AutoModelForCausalLM.from_pretrained(path)
        pmodel = PeftModel.from_pretrained(model, 'finetuneLLM/finetuned_models/samsumBad-7b-fp16-peft-seed-42/',torch_dtype=torch.float16) #load peft model

        SafeLoRAConfig.base_model_path = './LLM_Models/llama-2-7b-hf/'  #you should modify the path
        SafeLoRAConfig.aligned_model_path = './LLM_Models/llama-2-7b-chat-fp16/' #you should modify the path

        safelora = SafeLoRA(pmodel, SafeLoRAConfig)

        Finally, you can get the projected model by "safelora.model".
        """
        super().__init__()
        self.ft_model = ft_model
        self.config = config
        
        # self.peft_config = ft_model.peft_config["default"]

        if self.config.target == "lora":
            self.proj_modules = ["q_proj", "v_proj"]
        else:
            self.proj_modules = "full"

        # proj_modules = "all"

        self.model_ori = copy.deepcopy(ft_model)
        project_matrix = self.get_aligned_matrix()
        if self.config.select_layers_type == 'threshold':
            self.model, _ = self.projected_weighted(project_matrix, self.config.threshold, show_info=True)
        elif self.config.select_layers_type == 'number':
            model, cos = self.projected_weighted(project_matrix, 0.3, show_info=False)
            thrs = numpy.sort(cos)[:self.config.num_proj_layers][-1]
            self.model, _ = self.projected_weighted(project_matrix, thrs, show_info=True)
        else:
            raise ValueError("The method of select_layer_type should be threshold or number.")

    def get_aligned_matrix(self):
        """
        Get projected matrix by following the config (target_modules) from the peft model.
        The dimensions between the base model's weights and the aligned model's weights should be the same.
        """
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        unaligned_model = AutoModelForCausalLM.from_pretrained(
            self.config.unaligned_model_path,
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        v = []
        # proj_modules = list(self.peft_config.target_modules)
        for (b_name, b_param) , (a_name, a_param) in zip (unaligned_model.named_parameters(), self.base_model.named_parameters()):
            if self.proj_modules == "full" or any(module in a_name for module in self.proj_modules):
                assert b_param.shape == a_param.shape, "The dimensions of the base model's weight should be the same with the aligned model's weight."
                try:
                    vec = a_param - b_param
                    vec = vec.to(self.config.devices)
                    

                    if len(vec.shape) == 1:
                        vec = vec.unsqueeze(dim = 1)
                        vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                        # vec
                    else:
                        vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                    v.append((vec).detach().cpu())
                except:
                    pass
                    import pdb; pdb.set_trace()
        return v

    def projected_weighted(self, project_matrix, thrs_cos, show_info=False):
        v = project_matrix
        idx = 0
        i = 0
        dis = []
        skip_count = 0
        cos_total = []
        for (name, param),(base_name, base_param) in zip(self.ft_model.named_parameters(), self.base_model.named_parameters()):
            pass
            if self.proj_modules == "all" or any(module in name for module in self.proj_modules):
                delta_W = param.data - base_param.data
                # idx += 1
                try:

                    P = v[idx].to(param.device)

                    if len(delta_W.shape) == 1:
                        # delta_W = delta_W.unsqueeze(dim = 1)
                        proj_W = torch.mm(P, delta_W.unsqueeze(dim = 1)).squeeze()
                    else:
                        proj_W = torch.mm(P, delta_W)
                    cos = numpy.round(torch.nn.functional.cosine_similarity(proj_W.reshape(1,-1), delta_W.reshape(1,-1)).item(),5)
                    cos_total.append(cos)
                except:
                    pass
                    import pdb; pdb.set_trace()
                if cos <=  thrs_cos:
                    i+=1
                    param.data =  base_param.data + proj_W
                    print("safe project layer: {}".format(name))
                else:
                    # param.data = param_ori
                    skip_count += 1
                    pass
                # dist = 1 / (1+torch.norm(param.data.reshape(1,-1)-W.reshape(1,-1)))

                # dis.append(dist.item())
                idx += 1

        if show_info:
            print(f"{i} layers are projected, cosine threshold is {thrs_cos}, and Pdst is {numpy.mean(dis)} (> 0.8 is better).")
            print("skip_count: {}".format(skip_count))
        return self.ft_model, cos_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='The fine-tuned model , e.g. local model path.', type=str, required=True)
    parser.add_argument('--base_model', help='The base model for alignment, e.g. local model path.', type=str, required=True)
    parser.add_argument('--unaligned_model', help='The unaligned model path to compare against the aligned model.', type=str, required=True)
    parser.add_argument('--save_path', help='Path where the model results will be saved. Default is "evaluate/results".', type=str, required=False, default='evaluate/results')
    parser.add_argument('--save_name', help='The name for the saved model results. If not provided, defaults to model name.', type=str, required=False, default=None)
    parser.add_argument('--target', help='Operation target; options include "full" for full evaluation or "lora" for LoRA-specific evaluations.', type=str, required=True, default="full")
    parser.add_argument('--select_layers_type', help='Method to select projection layers; options are "threshold" and "number".', type=str, required=False, default="threshold")
    parser.add_argument('--threshold', help='Cosine similarity threshold for layer selection, relevant for "threshold" method.', type=float, required=False, default=0.5)

    args = parser.parse_args()

    safe_lora_config = SafeLoRAConfig()
    safe_lora_config.unaligned_model_path   = args.unaligned_model
    safe_lora_config.base_model_path        = args.base_model
    safe_lora_config.select_layers_type     = args.select_layers_type
    safe_lora_config.threshold              = args.threshold
    safe_lora_config.target                 = args.target

    ft_model = AutoModelForCausalLM.from_pretrained(args.model)

    safe_lora_runner = SafeLoRA(ft_model=ft_model, config=safe_lora_config)

    ft_model = safe_lora_runner.model

    # 加载预训练模型和tokenizer

    # model_name = args.model  # 选择合适的模型名称
    
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    save_name = args.save_name
    print(f"model and tokenizer will be saved on: {save_name}")
    ft_model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)
    
    # torch.save(task_vector, save_name)
if __name__ == "__main__":
    main()
