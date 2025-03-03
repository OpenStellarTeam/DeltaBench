# Python standard library imports
import asyncio
import json
import os
import random
import re
import sys
import time
import traceback
import argparse
from collections import Counter
from multiprocessing import Pool
import aiohttp
import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from tqdm.asyncio import tqdm
from utils.prompt import *


# os.environ["OPENAI_API_KEY"] = "replace your key here"
# os.environ["OPENAI_BASE_URL"] = "replace your key here"
os.environ["OPENAI_API_KEY"] = "9d0f097eae5d256de297859404694af7"
os.environ["OPENAI_BASE_URL"] = "https://idealab.alibaba-inc.com/api/openai/v1"


def extract_first_number(s):
    """
    Extract the first number from a string using regex.
    Args:
        s (str): Input string
    Returns:
        str: First number found in string, or None if no number exists
    """
    match = re.search(r'\d+', s)
    return match.group() if match else None


def parse_output(critic, all_error_section_indexs):
    """
    Parse the model's critique output and calculate evaluation metrics.
    
    Args:
        critic (str): Model's critique text
        all_error_section_indexs (list): List of all error section indices
        
    Returns:
        dict: Dictionary containing parsing results and evaluation metrics
    """
    # Initialize variables
    judge = -1
    parsing_success = 0
    precision = recall = f1_score = 0
    tp_step = fp_step = fn_step = 0
    try:
        # Extract conclusion and determine if there are errors
        result = critic.split("Error Section Number:")[0].split("Conclusion:")[-1].strip()
        has_errors = "yes" in result.lower()
        
        if has_errors:
            # Parse error sections and explanations
            model_judges = critic.split("Error Section Number:")[1:]
            error_sections_nums = []
            explanation = []
            
            for cur_error in model_judges:
                # Extract error number and explanation
                cur_error_number = extract_first_number(cur_error.split("Explanation:")[0].strip())
                cur_error_number = int(cur_error_number) if cur_error_number else -1
                cur_error_explanation = cur_error.split("Explanation:")[-1].strip()
                
                error_sections_nums.append(cur_error_number)
                explanation.append(cur_error_explanation)
            
            judge = 1
            parsing_success = 1
        else:
            judge = 0
            error_sections_nums = []
            explanation = []
            parsing_success = 1

        # Filter error sections based on max label
        max_label_error_section = max(all_error_section_indexs)
        error_sections_nums = [x for x in error_sections_nums if x <= max_label_error_section]

        # Calculate true positives, false positives, and false negatives
        true_positives = len(set(error_sections_nums) & set(all_error_section_indexs))
        false_positives = len(set(error_sections_nums) - set(all_error_section_indexs))
        false_negatives = len(set(all_error_section_indexs) - set(error_sections_nums))
        
        # Calculate precision, recall and F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        tp_step = true_positives
        fp_step = false_positives
        fn_step = false_negatives

        return {
            "result": result,
            "error_sections_nums": error_sections_nums,
            "explanation": explanation,
            "all_info": critic,
            "parsing_success": parsing_success,
            "judge": judge,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp_step": tp_step,
            "fp_step": fp_step,
            "fn_step": fn_step,
        }
    
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing the model output: {e}")
        return {
            "all_info": critic,
            "parsing_success": 0,
            "judge": judge,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }



def call_model(messages, modelname):
    """Call Model API with retry mechanism.
    
    Args:
        messages (list): List of message dictionaries
        modelname (str): Name of the model to use
        
    Returns:
        tuple: (model output, token usage information)
    """
    k = 3
    output = ""
    token_info = {}
    while(k > 0):
        k -= 1
        try:
            client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ["OPENAI_BASE_URL"],
            )
            completion = client.chat.completions.create(
                model=modelname,
                messages=messages,
                top_p=0.8,
                temperature = 1
            )
            output = completion.choices[0].message.content
            total_tokens = completion.usage.total_tokens
            prompt_token = completion.usage.prompt_tokens
            completion_token = completion.usage.completion_tokens
            token_info = {
                "total_tokens": total_tokens,
                "prompt_token": prompt_token,
                "completion_token": completion_token
            }
            if output != None and output != "":
                break
        except Exception as e:
            print(e)
            continue
    return output, token_info


def write_to_file(info, new_file):
    if not isinstance(info, str):
        info = json.dumps(info, ensure_ascii=False)
    with open(new_file, 'a', encoding='utf-8') as fin:
        fin.write(info + '\n')


def process_line(args_data):
    line, args = args_data
    question = line['question']
    if "sections_content" in line:
        model_output = line['sections_content']
    else:
        model_output = line['section_content']

    idea_error_section_numbers = line['reason_unuseful_section_numbers']
    error_section_numbers = line['reason_error_section_numbers']
    
    all_section_indexs = idea_error_section_numbers + error_section_numbers
    all_section_indexs = list(set(all_section_indexs))
    parsing_success = line.get("parsing_success", 0)
   
    messages = []
    prompt = critic_judge_prompt.replace("{{question}}", question).replace("{{model_output}}", model_output)
    messages.append({"role": "user", "content": prompt})
    line['messages'] = messages
    try:
        critic = line.get("critic", "")
        token_info = {}
        if critic == "":
            output,token_info = call_model(messages, args.call_modelname)
            critic = output 
        print(critic)
        line['critic'] = critic
        line['token_info'] = token_info
            
        if isinstance(critic, str) and critic == "":
            line['parsing_success'] = 0
            line['info'] = "output is None"
            write_to_file(line, args.new_file)
            return 0
        
        info = parse_output(critic, all_section_indexs)
        line.update(info)
        write_to_file(line, args.new_file)
        return info['parsing_success']
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        write_to_file(line, args.new_file)
        return 0


def deal_down_data(origin_file, new_file):
    done = {} 
    if os.path.exists(new_file):
        with open(new_file, "r", encoding='utf-8') as fin:
            done_lines = fin.readlines()
            for line in done_lines:
                data = json.loads(line)
                critic = data.get("critic", "")
                if critic != "":
                    question = data['question']
                    done[question] = data
        new_file = new_file.replace(".jsonl", "_1.jsonl")
    data_new = []
    with open(origin_file, "r", encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            data = json.loads(line)
            if 'question' not in data:
                continue
            question = data['question']
            if question in done:
                data_new.append(done[question])
            else:
                data_new.append(data)
    return data_new
    
def calculate_accuracies_v2(group):

    total_questions = len(group)

    total_predicted_errors = group[group['judge'] == 1].shape[0]
    total_predicted_correct = group[group['judge'] == 0].shape[0]
    
    precision_macro = group['precision'].mean()
    recall_macro = group['recall'].mean()
    f1_score_macro = group['f1_score'].mean()
    
    
    sum_tp = group['tp_step'].sum()
    sum_fp = group['fp_step'].sum()
    sum_fn = group['fn_step'].sum()
    
    precision_micro = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    recall_micro = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
    
    return pd.Series({
        'recall_macro': recall_macro,
        'precision_macro': precision_macro,
        'f1_score_macro': f1_score_macro,
        'recall_micro': recall_micro, 
        'precision_micro': precision_micro,  
        'f1_micro': f1_micro,  
    })


def get_metrics(new_file):
    with open(new_file, "r", encoding='utf-8') as fin:
        lines = fin.readlines()
        datas = [json.loads(line) for line in lines]
        df = pd.json_normalize(datas)
        accuracy_df = calculate_accuracies_v2(df)

        overall_row = pd.DataFrame({
            'task_l1': ['Overall'],
            'recall_macro': [accuracy_df['recall_macro']],
            'precision_macro': [accuracy_df['precision_macro']],
            'f1_score_macro': [accuracy_df['f1_score_macro']],
            'recall_micro': [accuracy_df['recall_micro']],
            'precision_micro': [accuracy_df['precision_micro']],
            'f1_micro': [accuracy_df['f1_micro']],
        })

        accuracy_df = df.groupby('task_l1').apply(calculate_accuracies_v2).reset_index()
        final_df = pd.concat([overall_row, accuracy_df], ignore_index=True)
        final_df.to_csv(new_file.replace(".jsonl", ".csv"), index=False)
        
def evaluation(args, processes=10):
    fin =  open(args.new_file, "w", encoding='utf-8')
    # Parallel evaluation
    arg_list = []
    for data in data_new:
        arg_list.append((data, args))
    with Pool(processes=processes) as pool:
        results = list(tqdm(pool.imap(process_line, arg_list), total=len(data_new)))       
        correct = np.sum(np.array(results))
        print("success num: ", correct)   
    return correct
    

# ==============================
      
if __name__ == "__main__":
    start_time = time.perf_counter()   
    parser = argparse.ArgumentParser()
    parser.add_argument('--call_modelname', required=False, default=None, 
                        help='The path of the data to process.')
    parser.add_argument('--dataset', required=False, default=None)
    
    args = parser.parse_args()
    print(f"Model Name: {args.call_modelname}")
    print(f"dataset: {args.dataset}")

    origin_file = f"data/{args.dataset}.jsonl"
    new_file = f"evaluation/{args.dataset}_{args.call_modelname}.jsonl"
    
    # Get unmeasured data
    data_new = deal_down_data(origin_file, new_file)
    fin =  open(new_file, "w", encoding='utf-8')
    
    # Parallel evaluation
    args.new_file = new_file
    correct = evaluation(args, processes = 10)   
        
    # If the number of failed tests is too high, retry
    k = 0 
    all_num = int(len(data_new)*0.95)
    while correct < all_num and k < 3:
        k += 1
        print(f"fail num is {all_num - correct}, try again")
        start_time = time.perf_counter()    
        origin_file = new_file
        with open(origin_file, "r", encoding='utf-8') as fin:
            lines = fin.readlines()
            lines = [json.loads(line) for line in lines]
        new_file = f"{new_file}_{k}.jsonl"
        args.new_file = new_file
        correct = evaluation(args, processes = 1)    
    # Calculate relevant evaluation indicators
    get_metrics(new_file)
    
    
    end_time = time.perf_counter()
    execution_time_ms = (end_time - start_time) / 60
    print(f"time: {execution_time_ms} mins")
    