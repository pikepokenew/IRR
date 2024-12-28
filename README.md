# IRR Method

## Restore safety to fine-tuned models

```bash
      method="alpha_test_v2"
      decorate=1
      new_model_name=llama2_fft_CodeAlpaca-20k_v5_task_${method}_sparsity_${sparsity}_full_blocksize_${bs}_w_decorate
      sft_model_name=llama2_fft_CodeAlpaca-20k_v5_task
      safe_FIM_name="~/IRR/FIMs/BeaverTails_1k_Llama-2-7b-chat-hf_sys_1Llama-2-7b-chat-hf_all_fisher_matrix.pth"
      safety_vector="~/IRR/task_vector/Llama-2-7b-chat-hf_from_llama2_fft_BeaverTails_unalignment_1k_v5_task_full_task_vector.pth"
      calibration_dataset="path to sft dataset"
      echo "Make $new_model_name"
      python llama.py  --model ${sft_model_path}  --base_model ${base_model} --dataset $calibration_dataset --nsamples 1000 --safe_FIM_path ${safe_FIM_name}  --safety_vector ${safety_vector}  --save saved_models/${new_model_name}  --sparsity $sparsity --blocksize $bs --score $method --decorate $decorate
```


## Evaluating Model Safety Against Harmful Instructions

This project evaluates the safety of models when confronted with harmful instructions.

Use the following command to generate model responses:

```bash
python evaluate/generate_responses_v2.py \
  --model $model_name \
  --dataset evaluate/harmful_questions/${dataset_name}.json \
  --save_path evaluate/results \
  --save_name ${dataset_name}/${model_name}_sys_${system_prompt}.json \
  --need_system_prompt $system_prompt
```

Evaluation Using MD-Judge
Note: MD-Judge, a separate moderation model, must be downloaded additionally.

To evaluate model responses with MD-Judge, run:

```bash
python evaluate/moderation_as_judge_v2.py \
  --response_file evaluate/results/${dataset_name}/${model_name}_sys_${system_prompt}.json \
  --save_path evaluate/results \
  --batch_size $batch_size \
  --moderation "MD-Judge"
```
Evaluating Mathematical Abilities on GSM8K
To assess the model's mathematical capabilities, execute the following command:

```bash
python eval_gsm8k_zero_shot_v2.py \
  --model $model_name \
  --use_cot_prompt \
  --batch_size $batch_size \
  --system_prompt $system_prompt
```
Generating and Evaluating Results on HumanEval
Use the commands below to generate and evaluate results for HumanEval:

```bash
output_variable=$(python evaluate/generate_human_eval_v2.py \
  --model $base_model \
  --batch_size 16 \
  --use_chat_template \
  --use_system_prompt \
  --num_samples_per_task $num_samples)
echo $output_variable
file_path=$(echo "$output_variable" | awk -F 'Completed, please check ' '{print $2}')
conda activate codex
echo $file_path
evaluate_functional_correctness $file_path
conda deactivate
```
Evaluating Chinese Language Abilities on MMMLU-ZH
To evaluate the model's performance in Chinese, run the following command:

```bash
python llama3.py \
  --model_name_or_path $base_model \
  --data_dir ../data/MMMLU_ZH_CN \
  --num_few_shot 0 \
  --use_system_prompt $system_prompt \
  --over_write 1 \
  --language "ZH_CN"
```
