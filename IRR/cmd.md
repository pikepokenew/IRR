##
```
method="safe_fisher"
decorate=1
new_model_name=llama3_peft_mathinstruct_adapter_ep1_TIES_${method}_sparsity_${sparsity}_q_proj_v_proj_blocksize_${bs}
sft_model_name=~/LLaMA-Factory/models/llama3_peft_mathinstruct_adapter_ep1
safe_FIM_name="~/ReAlign/FIMs/BeaverTails_1k_Meta-Llama-3-8B-Instruct_q_proj_v_proj_fisher_matrix.pth"
safety_vector="~/ReAlign/task_vector/Meta-Llama-3-8B-Instruct_from_llama3_peft_BeaverTails_unalignment_adapter_q_proj_v_proj_task_vector.pth"
calibration_dataset="~/resta/sft/data/MathInstruct.json"
echo "Make $new_model_name"

python llama.py  --model ~/LLaMA-Factory/models/llama3_peft_mathinstruct_adapter_ep1  --base_model /home/share/models/Meta-Llama-3-8B-Instruct --dataset ~/resta/sft/data/MathInstruct.json --nsamples 1000 --safe_FIM_path ~/ReAlign/FIMs/BeaverTails_1k_Meta-Llama-3-8B-Instruct_q_proj_v_proj_fisher_matrix.pth  --safety_vector ~/ReAlign/task_vector/Meta-Llama-3-8B-Instruct_from_llama3_peft_BeaverTails_unalignment_adapter_q_proj_v_proj_task_vector.pth  --save saved_models/llama3_peft_mathinstruct_adapter_ep1_TIES_safe_fisher_sparsity_0.96_q_proj_v_proj_blocksize_128  --sparsity 0.96 --blocksize 128 --score safe_fisher --decorate 1 --need_system_prompt 0 --true-sequential


python llama.py  --model ~/LLaMA-Factory/models/llama3_peft_mathinstruct_adapter_ep1  --base_model /home/share/models/Meta-Llama-3-8B-Instruct --dataset ~/resta/sft/data/MathInstruct.json --nsamples 1000 --safe_FIM_path ~/ReAlign/FIMs/BeaverTails_1k_Meta-Llama-3-8B-Instruct_q_proj_v_proj_fisher_matrix.pth  --safety_vector ~/ReAlign/task_vector/Meta-Llama-3-8B-Instruct_from_llama3_peft_BeaverTails_unalignment_adapter_q_proj_v_proj_task_vector.pth  --save saved_models/llama3_peft_mathinstruct_adapter_ep1_TIES_safe_fisher_sparsity_0.9999_q_proj_v_proj_blocksize_128  --sparsity 0.9999 --blocksize 128 --score alpha_test --decorate 0 --need_system_prompt 0 --true-sequential

python llama.py  --model ~/resta/saved_models/llama2_fft_CodeAlpaca-20k_v5_task  --base_model ~/local_models/Llama-2-7b-chat-hf --dataset ~/resta/sft/data/CodeAlpaca-20k.json --nsamples 1000 --safe_FIM_path ~/ReAlign/FIMs/BeaverTails_1k_Llama-2-7b-chat-hf_sys_1Llama-2-7b-chat-hf_all_fisher_matrix.pth  --safety_vector ~/ReAlign/task_vector/Llama-2-7b-chat-hf_from_llama2_fft_BeaverTails_unalignment_1k_v5_task_full_task_vector.pth  --save saved_models/llama2_fft_CodeAlpaca-20k_v5_task_alpha_test_sparsity_0.7_full_blocksize_128  --sparsity 0.7 --blocksize 128 --score alpha_test_v2 --decorate 0 --need_system_prompt 1 --true-sequential
      
```
