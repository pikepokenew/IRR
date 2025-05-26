#!/bin/bash
#SBATCH -J Eval_Safety
#SBATCH -o logs_and_outputs/Eval_Safety_test-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 24:00:00     
#SBATCH --mem=96gb
#SBATCH --gres=gpu:a100x:1
#SBATCH -w gpu05

source /home/dwu/miniconda3/etc/profile.d/conda.sh
run_env=irr
conda activate $run_env

nvidia-smi

screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:10233 && export https_proxy=http://127.0.0.1:10233 && export all_proxy=http://127.0.0.1:10233

stage1="true"  # 控制是否执行 Safety 测试阶段
stage2="false"  # 控制是否执行 JailBreak 测试阶段
gsm8k_stage="false"  # 控制是否执行 GSM8K 测试阶段
human_eval_stage="true"  # 控制是否执行 Human_eval 测试阶段
mmmlu_zh_stage="false"  # 控制是否执行 MMLU-ZH 测试阶段
system_prompt=1
# calibration_nsamples=64
# calibration_nsamples_list=("8" "64" "128" "1000")
# calibration_nsamples_list=("8" "64")
# calibration_nsamples_list=("128")
calibration_nsamples_list=("1000")
cd ~/IRR/
model_list=()
# blocksize=("4096" "256" "16" "1")

blocksize=("128")
# sparsity_list=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
sparsity_list=("0.0")

safety_dataset_name_list=("catqa_english" "catqa_chinese" "catqa_vietnamese" "HEx-PHI" "Salad-base_set_sub_v1")

jailbreak_dataset_name_list=("Salad-attack_enhanced_set")

sft_model_path="/home/dwu/resta/saved_models"
# sft_model_path="~/LLaMA-Factory/models"

for calibration_nsamples in "${calibration_nsamples_list[@]}"; do
   for sparsity in "${sparsity_list[@]}"; do
      for bs in "${blocksize[@]}"; do
         ##### For LoRA #####
         # ############### CodeAlpaca-20k ###############
         method="IRR"
         recalibrate=1
         new_model_name=llama2_peft_CodeAlpaca-20k_v5_adapter_${method}_sparsity_${sparsity}_q_proj_v_proj_blocksize_${bs}
         sft_model_name=llama2_peft_CodeAlpaca-20k_v5_adapter
         calibration_dataset="data/CodeAlpaca-20k.json"
         safety_vector="task_vector/Llama-2-7b-chat-hf_from_llama2_peft_BeaverTails_unalignment_1k_v5_adapter_q_proj_v_proj_task_vector.pth"
         safe_FIM="FIMs/BeaverTails_1k_Llama-2-7b-chat-hf_sys_1_Llama-2-7b-chat-hf_q_proj_v_proj_fisher_matrix_v2.pth"
         echo "Make $new_model_name"

         python IRR/llama.py  --model ${sft_model_path}/${sft_model_name}  --base_model ~/local_models/Llama-2-7b-chat-hf --dataset $calibration_dataset  --nsamples $calibration_nsamples  --true-sequential --safe_FIM_path $safe_FIM  --safety_vector $safety_vector  --save saved_models/${new_model_name}  --sparsity $sparsity --blocksize $bs --method $method --recalibrate $recalibrate --need_system_prompt $system_prompt
         
         ###### For Full Fine tuning Model #####
         # method="IRR"
         # recalibrate=1
         # new_model_name=llama2_fft_gsm8k_v5_task_TIES_${method}_sparsity_${sparsity}_full_blocksize_${bs}_nsamples_${calibration_nsamples}
         # sft_model_name=llama2_fft_gsm8k_v5_task
         # safe_FIM_name="FIMs/BeaverTails_1k_Llama-2-7b-chat-hf_sys_1_Llama-2-7b-chat-hf_all_fisher_matrix.pth"
         # safety_vector="task_vector/Llama-2-7b-chat-hf_from_llama2_fft_BeaverTails_unalignment_1k_v5_task_full_task_vector.pth"
         # calibration_dataset="data/CodeAlpaca-20k.json"

         # echo "Make $new_model_name"
         # python llama.py  --model ${sft_model_path}/${sft_model_name}  --base_model ~/local_models/Llama-2-7b-chat-hf --dataset $calibration_dataset --nsamples ${calibration_nsamples} --safe_FIM_path ${safe_FIM_name}  --safety_vector ${safety_vector}  --save saved_models/${new_model_name}  --sparsity $sparsity --blocksize $bs --method $method --recalibrate $recalibrate
         # model_list+=("$new_model_name")
         
         model_list+=("$new_model_name")
      done
   done
done



cd ~/IRR/

for model_name in "${model_list[@]}"; do
   cd ~/IRR/

   base_model=saved_models/${model_name}
   # dataset_name_list=("catqa_english" "catqa_chinese" "catqa_vietnamese" "harmfulqa" "dangerousqa" "adversarialqa")

   if [[ "$stage1" == "true" ]]; then
      echo "--------------Test Safety--------------"
      for dataset_name in "${safety_dataset_name_list[@]}"; do
         
         echo "Now evaluation on: ${dataset_name}"
         conda activate $run_env
         python evaluate/generate_responses.py --model $base_model --dataset evaluate/harmful_questions/${dataset_name}.json --save_path evaluate/results --save_name ${dataset_name}/${model_name}_sys_${system_prompt}.json --need_system_prompt $system_prompt

         python evaluate/moderation_as_judge.py --response_file evaluate/results/${dataset_name}/${model_name}_sys_${system_prompt}.json --save_path evaluate/results --moderation "MD-Judge"
         conda deactivate
         # python evaluate/gpt4_as_judge.py --response_file evaluate/results/${dataset_name}/${model_name}_sys_${system_prompt}.json --save_path evaluate/results --pool_size 10
      done
   fi

   if [[ "$stage1_2" == "true" ]]; then
      echo "--------------Test JailBreak--------------"
      for dataset_name in "${jailbreak_dataset_name_list[@]}"; do
         
         echo "Now evaluation on: ${dataset_name}"
         conda activate $run_env
         python evaluate/generate_responses.py --model $base_model --dataset evaluate/harmful_questions/${dataset_name}.json --save_path evaluate/results --save_name ${dataset_name}/${model_name}_sys_${system_prompt}.json --need_system_prompt $system_prompt

         python evaluate/moderation_as_judge.py --response_file evaluate/results/${dataset_name}/${model_name}_sys_${system_prompt}.json --save_path evaluate/results --moderation "MD-Judge"
         conda deactivate
      done
   fi

   if [[ "$gsm8k_stage" == "true" ]]; then
      echo "--------------Test GSM8K--------------"
      cd ../IRR/evaluate

      echo "*** WITH SYSTEM PROMPT ***"
      conda activate $run_env
      python eval_gsm8k_zero_shot.py --model $base_model --use_cot_prompt --system_prompt $system_prompt
      conda deactivate
   fi

   if [[ "$human_eval_stage" == "true" ]]; then
      echo "--------------Test Human Eval--------------"
      cd ~/IRR/
      # num_samples_per_task=("1" "10")
      num_samples_per_task=("10")
      for num_samples in "${num_samples_per_task[@]}"; do
         conda activate $run_env
         output_variable=$(python evaluate/generate_human_eval.py --model $base_model --use_chat_template --use_system_prompt --num_samples_per_task $num_samples)
         echo $output_variable
         file_path=$(echo "$output_variable" | awk -F 'Completed, please check ' '{print $2}')
         conda deactivate
         conda activate codex
         echo $file_path
         evaluate_functional_correctness $file_path
         conda deactivate
      done
   fi

   if [[ "$mmmlu_zh_stage" == "true" ]]; then
      echo "-------------- Test MMMLU_ZH_CN --------------"
      cd ~/IRR/
      system_prompt_sign=("1")
      for use_system_prompt in "${system_prompt_sign[@]}"; do
         conda activate $run_env
         cd ~/IRR/CMMLU/src

         python llama3.py --model_name_or_path $base_model --data_dir ../data/MMMLU_ZH_CN --num_few_shot 0 --use_system_prompt $system_prompt --over_write 1 --language "ZH_CH"
         conda deactivate
         # fi
      done
   fi

   # cd ~/IRR/
   # python evaluate/collect_results.py --response_file ${model_name}_sys_${system_prompt}

done