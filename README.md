# IRR

## Overview

IRR is a post-hoc method designed to restore safety performance in fine-tuned language models while preserving their downstream task capabilities. This repository contains the code to reproduce the key results from the paper "[Separate the Wheat from the Chaff: A Post-Hoc Approach to Safety Re-Alignment for Fine-Tuned Language Models](https://arxiv.org/abs/2412.11041)".

## Key Features
This codebase provides the following scripts and implementations:

* Safety Vector and Fisher Information Matrix Calculation:
  * ```compute_safety_vector.py```: Computes the safety vector for re-alignment.
  * ```compute_fisher_information_matrix.py```: Calculates the Fisher Information Matrix (FIM) as a safety importance score.

* Safety Re-Alignment with IRR:
  * ```llama.py```: Applies the IRR method to fine-tuned models for safety re-alignment.

* Model Generation and Evaluation:
  * ```generate_responses.py```: Generates responses from the model.
  * ```moderation_as_judge.py```: Evaluates the generated responses for safety.

## Getting Started
### Prerequisites

To use this codebase, you need:

* A base aligned model (e.g., LLaMA-2).
* A fine-tuned model (SFT model) derived from the base model.
* An unaligned model derived from the base model.

### ⚙️ Environment Setup
#### Set up a Conda environment:
```
conda create -n irr python=3.10
conda activate irr
```

#### Clone the repository and install the required dependencies:
```
git clone https://github.com/pikepokenew/IRR.git
cd IRR
pip install -r requirements.txt
```

### Usage

The following steps outline how to use IRR for safety re-alignment, using a LoRA fine-tuned model as an example.

#### Step 1: Compute Safety Vector

```
python src/compute_safety_vector.py --model ${ALIGNED_MODEL_PATH} --base_model ${UNALIGNED_MODEL_PATH} --target lora
```

#### Step 2: Compute Safety Importance Score (Fisher Information Matrix)

* 2.1 Generate Safety Response Data

```
python src/generate_responses.py --model ${ALIGNED_MODEL_PATH} --dataset ${HARMFUL_QUESTIONS_DATA} --need_system_prompt 1
```

* 2.2 Calculate Fisher Information Matrix
```
python src/compute_fisher_information_matrix.py --model ${ALIGNED_MODEL_PATH} --dataset ${SAFETY_RESPONSE_DATA} --need_system_prompt 1 --target lora
```

### Step 3: Run IRR for Safety Re-Alignment
```
python IRR/llama.py \
  --model ${FINETUNED_MODEL_PATH} \
  --base_model ${BASE_MODEL_PATH} \
  --dataset data/CodeAlpaca-20k.json \
  --nsamples 10 \
  --true-sequential \
  --safe_FIM_path ${SAFETY_IMPORTANCE_SCORE_PATH} \
  --safety_vector ${SAFETY_VECTOR_PATH} \
  --save ${SAVE_MODEL_PATH} \
  --sparsity 0.0 \
  --blocksize 128 \
  --method IRR \
  --recalibrate 1 \
  --need_system_prompt 1
```
For additional usage examples, refer to the ```scripts/eval_model_irr.sh``` script for configuration details.

### Key Parameters for llama.py
```
Below are the key parameters for the llama.py script used in the IRR process:

--safety_vector  
Type: String  
Description: Path to the safety vector data used to guide pruning (optional).  
Default: None

--true-sequential`
Type: Boolean
Description: Whether to run the model in true sequential mode (only prunes `self_attn.v_proj` and `self_attn.q_proj`).
Default: `False`

--recalibrate  

Type: Integer  
Description: Whether to recalibrate the model (0 for no, 1 for yes).  
Default: 1

--remove_more  

Type: Integer  
Description: Whether to remove more weights. When used with --method IRR, it corresponds to the IRR_more method described in the paper (0 for no, 1 for yes).  
Default: 0

--need_system_prompt  

Type: Integer  
Description: Whether a system prompt is required (0 for no, 1 for yes).  
Default: 1

--method  

Type: String  
Description: The pruning method to use (e.g., IRR).  
Default: "IRR"
```

## References
* https://github.com/ist-daslab/sparsegpt


## Citation
If you found this work useful, please consider citing:
```
@article{wu2024separate,
  title={Separate the Wheat from the Chaff: A Post-Hoc Approach to Safety Re-Alignment for Fine-Tuned Language Models},
  author={Wu, Di and Lu, Xin and Zhao, Yanyan and Qin, Bing},
  journal={arXiv preprint arXiv:2412.11041},
  year={2024}
}
```
