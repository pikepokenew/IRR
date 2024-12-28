## cmd
```
python llama3.py --model_name_or_path ~/resta/saved_models/llama2_fft_alpaca_gpt4_zh_v5_task --data_dir ../data/MMMLU_ZH_CN_v2 --num_few_shot 0 --use_system_prompt 1 --over_write 1 --language "ZH_CH_v5"


python llama3.py --model_name_or_path ~/resta/saved_models/llama2_fft_alpaca_gpt4_zh_v5_task_wise_ft_0.9  --data_dir ../data/MMMLU_ZH_CN_v2 --num_few_shot 0 --use_system_prompt 1 --over_write 1 --language "ZH_CH_v5" --cot --tensor_parallel_size 2
```