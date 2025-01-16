import os
import torch
import numpy as np
import argparse
from mp_utils import choices, format_example, gen_prompt, softmax, get_results
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import re
import pandas as pd
from vllm import LLM, SamplingParams
import json
import sys
sys.path.append("/home/xxx/ReAlign")
from src.safe_decoding import SafeDecoding

llama_2_system_prompt = '''You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.'''

XFINDER_PROMPT_TEMPLATE = {
    "xFinder-qwen1505":
        """<|System|>:{system}
<|User|>:{input}
<|Bot|>:""",
    "xFinder-llama38it":
        """<|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
}
xfinder_query_template  = 'Question: """{question}"""\n\nOutput sentences: """{llm_output}"""\n\nAnswer range: {standard_answer_range}\n\nKey extracted answer: '


XFINDER_SYSTEM_PROMPT = "You are a help assistant tasked with extracting the precise key answer from given output sentences."


def run_eval(model, tokenizer, eval, args, is_pipline):
    if model and not is_pipline:
        model.eval()
    subjects=sorted([f.split(".csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test/"))])
    args.save_dir = f"{args.save_dir}_{args.num_few_shot}_shot"
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.use_system_prompt == 0:
        system_prompt = None
    else:
        system_prompt = llama_2_system_prompt

    if args.subjects == "all":
        pass
        subject_list = "all"
    else:
        subject_list = args.subjects.split(",")
        pass

    total_results = {}
    for subject in subjects:
        if subject_list != "all" and subject not in subject_list:
            continue
        out_file = os.path.join(args.save_dir, f"results_{subject}.csv")
        if os.path.exists(out_file) and args.over_write == False:  # If result file exist, skip this subject
            continue
        # dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + ".csv"), header=0, index_col=0)
        dev_df = None
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + ".csv"), header=0, index_col=0)

        results = eval(model=model,
                    tokenizer=tokenizer,
                    subject=subject,
                    dev_df=dev_df,
                    test_df=test_df,
                    num_few_shot=args.num_few_shot,
                    max_length=args.max_length,
                    cot=args.cot if 'cot' in args else False,
                    system_prompt = system_prompt,
                    language = args.language)
        # test_df['prediction'] = preds
        # if 'with_conf' in args and args.with_conf:
        #     test_df['conf'] = confs
        total_results[subject] = results
        # test_df.to_csv(out_file, header=None)

    return total_results
    # print result
    # get_results(args.save_dir)


def is_eval_success(args) -> bool:
    """judege if eval task is success by checking the result dir"""
    subjects = sorted(
        [f.split(".csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test/"))]
    )
    abs_save_dir = f"{args.save_dir}_{args.num_few_shot}_shot"
    if not os.path.exists(abs_save_dir):
        return False
    for subject in subjects:
        out_file = os.path.join(abs_save_dir, f"results_{subject}.csv")
        if not os.path.exists(out_file):
            # If any result file NOT exist, the eval isn't finished
            return False
    return True

MMMLU_multi_choice_tempalte = '''Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}'''


multi_choice_tempalte = {
    "JA_JP": '''以下の選択式問題に答えてください。回答の最後の行は次の形式で記載してください：「Answer: $LETTER」（引用符なし）。ここで、LETTERはABCDのいずれかです。回答する前に、ステップごとに考えてください。
{Question}

A) {A}
B) {B}
C) {C}
D) {D}''',
    "HI_IN": '''निम्नलिखित बहुविकल्पीय प्रश्न का उत्तर दें। आपके उत्तर की अंतिम पंक्ति निम्नलिखित प्रारूप की होनी चाहिए: 'Answer: $LETTER' (उद्धरण चिह्नों के बिना) जहां LETTER ABCD में से एक है। उत्तर देने से पहले चरण-दर-चरण सोचें।
{Question}

A) {A}
B) {B}
C) {C}
D) {D}''',
    "EN": MMMLU_multi_choice_tempalte,
    "ZH_CH_v2": '''
请回答以下选择题。你的回答的最后一行应当以下面的格式书写：'答案：$LETTER' （不带引号），其中字母可以是A、B、C或者D。在回答之前请逐步思考。
{Question}

A) {A}
B) {B}
C) {C}
D) {D}''',
    "ZH_CH_v3": MMMLU_multi_choice_tempalte,
    "ZH_CH_v4": '''
请回答以下选择题。你的回答应当以下面的格式书写：'答案：$LETTER' （不带引号），其中字母可以是A、B、C或者D。请直接给出正确选项。
{Question}

A) {A}
B) {B}
C) {C}
D) {D}'''.strip(),
    "ZH_CH_v5": '''请回答以下选择题，请直接给出正确答案的选项。
问题：{Question}
选项：
A) {A}
B) {B}
C) {C}
D) {D}
答案是：'''.strip()
}

JA_JP_multi_choice_tempalte = '''以下の選択式問題に答えてください。回答の最後の行は次の形式で記載してください：「Answer: $LETTER」（引用符なし）。ここで、LETTERはABCDのいずれかです。回答する前に、ステップごとに考えてください。
{Question}

A) {A}
B) {B}
C) {C}
D) {D}'''
def eval_chat(
    model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot, system_prompt = None, language = "ZH_CN",
):
    """eval meta-llama/Meta-Llama-3-70B
    ref: https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct#use-with-transformers
    """
    pipeline=model
    cors = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]
    # import pdb; pdb.set_trace()

    prompts = []
    questions = []
    labels  = []
    option_list = []

    for i in tqdm(range(test_df.shape[0])):
        if language == "ZH_CN" or language == "ZH_CH_v5":
            prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
            question = gen_prompt(
                dev_df=dev_df,
                subject=subject,
                prompt_end=prompt_end,
                num_few_shot=num_few_shot,
                tokenizer=tokenizer,
                max_length=max_length,
                cot=cot,
            )
        elif language == "HI_IN" or language == "AR_XY" or language == "BN_BD" or language == "JA_JP" or language == "EN" or language == "ZH_CH_v2" or language == "ZH_CH_v3" or language == "ZH_CH_v4" or language == "ZH_CH_v5":
            # import pdb; pdb.set_trace()
            # question = MMMLU_multi_choice_tempalte.format(Question = test_df.iloc[i]['Question'],
            #                                               A = test_df.iloc[i]['A'],
            #                                               B = test_df.iloc[i]['B'],
            #                                               C = test_df.iloc[i]['C'],
            #                                               D = test_df.iloc[i]['D'],)
            question = multi_choice_tempalte[language].format(Question = test_df.iloc[i]['Question'],
                                                          A = test_df.iloc[i]['A'],
                                                          B = test_df.iloc[i]['B'],
                                                          C = test_df.iloc[i]['C'],
                                                          D = test_df.iloc[i]['D'],)
            
        questions.append(question)
        label = test_df.iloc[i, test_df.shape[1] - 1]
        messages = []
        if system_prompt != None and system_prompt != "" :
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt = True)
        
        # import pdb; pdb.set_trace()
        options = [
            ['A', test_df.iloc[i, 1]],
            ['B', test_df.iloc[i, 2]],
            ['C', test_df.iloc[i, 3]],
            ['D', test_df.iloc[i, 4]],
        ]
        
        
        prompts.append(prompt)
        labels.append(label)
        option_list.append(options)


    # outputs = model.generate(prompts, SamplingParams(
    #             temperature=0.6,
    #             top_p=0.9,
    #             max_tokens=max_length,
    #             n=1,
    #             stop=stop_words,
    # ))
    # outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
    # outputs_list = [output.outputs[0].text for output in outputs]

    results = []
    for prompt, question, label, options in zip(prompts, questions, labels, option_list):
        results.append({
            "question": question,
            "prompt":   prompt,
            # "output": output,
            "gold_label": label,
            "standard_answer_range": options
        })
    # import pdb; pdb.set_trace()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Meta-Llama-3-8B-Instruct")
    parser.add_argument("--expert_model", type=str, default="Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--method_name", type=str, default="safe_decoding")
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--use_system_prompt", type=int, default=1)
    parser.add_argument("--over_write", type=int, default=0)
    parser.add_argument("--language", type=str, default="ZH_CN")
    parser.add_argument("--subjects", type=str, default="all")
    parser.add_argument("--type", type=str, default="cot")
    parser.add_argument('--tensor_parallel_size', help='tensor_parallel_size', required=False, type=int, default=1)
    parser.add_argument('--first_m', help='first_m', required=False, type=int, default=2)
    # parser.add_argument("--use_system_prompt", type=int, default=1)
    args = parser.parse_args()

    print(f"\n\nconfiguration")
    print(f"*{'-'*10}*")

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print(f"*{'-'*10}*\n\n")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, padding_side="left"
    )



    if args.save_dir == None:
        data_name = args.data_dir.split("/")[-1]
        if args.cot == True:
            cot_flag = "_cot"
        else:
            cot_flag = ""
        save_dir = "../results/{}/{}_{}_sys_{}_{}_shot{}/".format(data_name, args.model_name_or_path.split("/")[-1], args.method_name, args.use_system_prompt, args.num_few_shot, cot_flag)

        
    else:
        save_dir = args.save_dir

    if not os.path.exists(save_dir):
    # 如果不存在，则创建文件夹
        os.makedirs(save_dir)
        # print(f"文件夹已创建：{save_dir}")
    else:
        pass
        # print(f"文件夹已存在：{save_dir}")

    print("all results will be saved at: {}".format(save_dir))
    if args.cot == True:
        max_new_tokens = 1024
    else:
        max_new_tokens = 256
    model = SafeDecoding(model_path=args.model_name_or_path, expert_model_path=args.expert_model,max_new_tokens=max_new_tokens,first_m=args.first_m)
    # model = LLM(args.model_name_or_path, tensor_parallel_size = args.tensor_parallel_size)
    # model=transformers.pipeline(
    #     "text-generation",
    #     model=args.model_name_or_path,
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     device_map="auto",
    # )

    
    total_results = run_eval(model, tokenizer, eval_chat, args, True)
    
    flatten_prompt_info = []
    prompts = []
    # import pdb; pdb.set_trace()
    ### LLM inference ###
    for subject, results in total_results.items():
        # import pdb; pdb.set_trace()
        for res in results:
            # import pdb; pdb.set_trace()
            res['subject'] = subject
            flatten_prompt_info.append(res)
            prompts.append(res['prompt'])

    print("----------------------------------------------------------------------------------")
    print("input:\n{}".format(prompts[0]))
    print("----------------------------------------------------------------------------------")
    # if args.language == "HI_IN":
    #     num_samples = 1000
    #     prompts             = prompts[:num_samples]
    #     flatten_prompt_info = flatten_prompt_info[:num_samples]
    # import random
    # num_samples = int(len(prompts) * 0.01)
    # random_indices = random.sample(range(len(prompts)), num_samples)
    # prompts = [prompts[i] for i in random_indices]
    # flatten_prompt_info = [flatten_prompt_info[i] for i in random_indices]

    # import pdb; pdb.set_trace()
    stop_words = []
    stop_words.append(tokenizer.eos_token)
    stop_words.append("<|eot_id|>")

    # outputs = model.generate(prompts, SamplingParams(
    #             temperature=0.0,
    #             top_p=0.9,
    #             max_tokens=max_new_tokens,
    #             n=1,
    #             stop=stop_words,
    # ))
    
    # outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
    # outputs_list = [output.outputs[0].text for output in outputs]
    
    outputs_list = model.generate(prompts,)

    total_results = {}
    for prompt_info, output in zip(flatten_prompt_info, outputs_list):
        # import pdb; pdb.set_trace()
        if total_results.get(prompt_info['subject'], None) == None:
            total_results[prompt_info['subject']] = []
        
        prompt_info['output'] = output
        total_results[prompt_info['subject']].append(prompt_info)


    ### Xfinder extract answer ###
    
    del model

    xfinder_prompt_list = []
    # xfinder_model_name = "xFinder-llama38it"
    xfinder_model_name = "xFinder-qwen1505"
    key_answer_type = "alphabet option"
    flatten_results = []
    for subject, results in total_results.items():
        for res in results:
            res['subject'] = subject
            flatten_results.append(res)
            # import pdb; pdb.set_trace()
            extract_formatted_query = xfinder_query_template.format(question = res['question'], llm_output = res['output'], standard_answer_range=str(res['standard_answer_range']))
            xfinder_prompt = XFINDER_PROMPT_TEMPLATE[xfinder_model_name].format(system=XFINDER_SYSTEM_PROMPT, input=extract_formatted_query)

            xfinder_prompt_list.append(xfinder_prompt)

            
    
    xfinder = LLM("/home/xxx/local_models/{}".format(xfinder_model_name), tensor_parallel_size = args.tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/xxx/local_models/xFinder-llama38it", trust_remote_code=True
    )
    stop_words = []
    stop_words.append(tokenizer.eos_token)
    stop_words.append("<|eot_id|>")
    outputs = xfinder.generate(xfinder_prompt_list, SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=32,
                n=1,
                stop=stop_words,
    ))
    outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
    # import pdb; pdb.set_trace()
    outputs_list = [output.outputs[0].text for output in outputs]

    total_result_info = {}
    for prompt_info, output in zip(flatten_results, outputs_list):
        
        if total_result_info.get(prompt_info['subject'], None) == None:
            total_result_info[prompt_info['subject']] = []
        
        info = {
            "subject":  prompt_info['subject'],
            "llm_output": prompt_info['output'],
            "question": prompt_info['question'],
            "pred":     output.strip(" \n"),
            "label":    prompt_info['gold_label'],
        }
        # prompt_info['output'] = output
        total_result_info[prompt_info['subject']].append(info)
        
        # pass

    total_acc = []
    total_acc_dict = {}
    for subject, result_info in total_result_info.items():
        out_file = os.path.join(save_dir, f"results_{subject}.json")
        cors = []
        all_preds = []
        # import pdb; pdb.set_trace()
        with open(f'{out_file}', 'w', encoding='utf-8') as f:
            json.dump(result_info, f, ensure_ascii=False, indent=4)
        # output_df = pd.DataFrame(result_info)
        # output_df.to_csv(out_file)
        # import pdb; pdb.set_trace()
        # pd.DataFrame(harmful_benchmrak_info)
        invalid_answer_count = 0
        for info in result_info:
            pred    = info['pred']
            label   = info['label']
            cors.append(pred == label)
            if pred == "[No valid answer]":
                invalid_answer_count += 1

            all_preds.append(pred)

        acc = np.mean(cors)
        print("Average accuracy {:.3f} - {}".format(acc, subject))
        print(
            "{} results, {} inappropriate formated answers.".format(
                len(cors), invalid_answer_count
            )
        )
        total_acc_dict[subject] = acc
        total_acc.append(acc)
    
    
    mean_acc = np.mean(total_acc)
    total_acc_dict['total_avg_acc'] = mean_acc
    print("Avg total acc: {}".format(mean_acc))
    acc_save_name = os.path.join(save_dir, "results_total_accuary.json")
    print("acc info will be saved at: {}".format(acc_save_name))
    with open(f'{acc_save_name}', 'w', encoding='utf-8') as f:
        json.dump(total_acc_dict, f, ensure_ascii=False, indent=4)

    



    
    # import pdb; pdb.set_trace()
    # if "instruct" in args.model_name_or_path.lower():
    #     model=transformers.pipeline(
    #         "text-generation",
    #         model=args.model_name_or_path,
    #         model_kwargs={"torch_dtype": torch.bfloat16},
    #         device_map="auto",
    #     )
    #     run_eval(model, tokenizer, eval_chat, args, True)
    # else:
    #     # Todo
    #     raise NotImplementedError(f"Direct inference with llama3_base_model is currently not supported.")


