import requests
import json
import traceback
import aiohttp
import asyncio
import json
import traceback
import random
import re
import ast
import sys
from time import sleep
from tqdm.asyncio import tqdm
from multiprocessing import Pool
import numpy as np  
from openai import OpenAI
import pandas as pd
import os
from collections import Counter
os.environ["OPENAI_API_KEY"] = ""


critic_judge_prompt = """\
I will provide you with a question and a robot’s answer, divided into several sections. Your task is to evaluate each section of the robot’s answer for any errors.

**Evaluation Criteria:**
- Evaluate each section independently. Assess each section based solely on the accuracy and logic within that section.
- Do not consider subjective elements such as redundancy or stylistic preferences as errors.
- Do not consider corrections or reflections made in later sections. Even if a later section acknowledges and fixes an earlier mistake, the original section must still be marked as erroneous.
- If a subsequent section contains an error caused by an earlier section's mistake, do not count it as a new error.

**Output Format:**
- If you think all sections of the robot's answer are correct, output in the following format:  
    Conclusion: no error
- If you think any section contains an error, output in the following format:  
    Conclusion: yes
    Error Section Number: [section number]
    Explanation: [explanation for the error in this section]
    Error Section Number: [section number]
    Explanation: [explanation for the error in this section]
    … (repeat for each erroneous section)  


**example:**
- Question: Two numbers, $x$ and $y$ are selected at random from the interval $(0,3)$.  What is the probability that a triangle with sides of length 1, $x$, and $y$ exists?
- Robot's Answer: 
section1:
So I've got this probability problem here. It says that two numbers, x and y, are selected at random from the interval (0,3), and I need to find the probability that a triangle with sides of length 1, x, and y exists. Okay, let's break this down.

section2:
First, I need to recall the triangle inequality theorem. It states that for any triangle with sides a, b, and c, the following must be true:
1. a + b > c
2. a + c > b
3. b + c > a

section3:
So, in this case, the sides are 1, x, and y. That means I need to satisfy all three inequalities:
1. 1 + x > y
2. 1 + y > x
3. x + y > 1
Alright, now, to find the probability, I need to find the ratio of the area where these inequalities hold to the total area from which x and y are selected.

section4:
Now, I need to find the area where all three inequalities are satisfied.
Let me consider each inequality one by one.
First inequality: 1 + x > y
This can be rewritten as y < 1 + x.
Second inequality: 1 + y > x
This can be rewritten as y > x - 1.
Third inequality: x + y > 1
This can be rewritten as y > 1 - x.
Now, since x and y are both positive (from (0,3)), 1 - x could be negative if x > 1. But since y is positive, y > 1 - x will always hold true when x < 1, and when x >=1, y > 1 - x is always true because y > 0 and 1 - x <=0. So, this inequality is always satisfied in the given range. Therefore, I don't need to worry about this inequality separately.

section5:
So, effectively, I only need to consider the first two inequalities:
1. y < 1 + x
2. y > x - 1
And, of course, x and y are between 0 and 3.

- judge result: 
Conclusion: yes
Error Section Number: 4
Explanation: The analysis of the inequality \( x + y > 1 \) is incorrect. The argument that "this inequality is always satisfied in the given range" is flawed. Specifically, for small values of \( x \) and \( y \), such as when both are close to zero, the inequality \( x + y > 1 \) does not always hold. The constraint \( x + y > 1 \) must still be explicitly considered in the probability calculation.

**Input:**
- Question: {{question}}
- Robot's Answer: {{model_output}}
- judge result: \
"""

def extract_first_number(s):
    # 使用正则表达式来查找字符串中的数字
    match = re.search(r'\d+', s)
    if match:
        return match.group()
    else:
        return None


def parse_output(critic, evaluation_label, all_error_section_indexs):
    model_judge_error_step = -1
    judge = -1
    parsing_success = 0
    precision = 0
    recall = 0
    f1_score = 0
    tp_step = 0 
    fp_step = 0  
    fn_step = 0
    try:
        result = critic.split("Error Section Number:")[0].split("Conclusion:")[-1].strip()
        if "yes" in result or "Yes" in result:
            model_judges = critic.split("Error Section Number:")[1:]
            error_sections_nums = []
            explanation = []
            for cur_error in model_judges:
                cur_error_number = cur_error.split("Explanation:")[0].strip()
                cur_error_number = extract_first_number(cur_error_number)
                try:
                    cur_error_number = int(cur_error_number)
                except:
                    cur_error_number = -1
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
        
        correct = 0
        if evaluation_label == judge:
            if evaluation_label == 0:
                correct = 1
            elif evaluation_label == 1:
                for error_sections_num in error_sections_nums:
                    if error_sections_num in all_error_section_indexs:
                        correct = 1
        else:
            correct = 0
            
        max_label_error_section = max(all_error_section_indexs)
        error_sections_nums = [x for x in error_sections_nums if x <= max_label_error_section]
        
        
        
            
        if evaluation_label == 1:
            true_positives = len(set(error_sections_nums) & set(all_error_section_indexs))
            false_positives = len(set(error_sections_nums) - set(all_error_section_indexs))
            false_negatives = len(set(all_error_section_indexs) - set(error_sections_nums))
            
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
            "correct": correct,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp_step": tp_step,  # 新增
            "fp_step": fp_step,  # 新增
            "fn_step": fn_step,  # 新增
        }
    
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing the model output: {e}")
        return {
            "all_info": critic,
            "parsing_success": 0,
            "judge": judge,
            "correct": correct,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }



def call_openai(messages, modelname):
    k = 3
    output = ""
    token_info = {}
    while(k > 0):
        k -= 1
        try:
            client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url="https://idealab.alibaba-inc.com/api/openai/v1",
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


def write_to_file(info):
    if not isinstance(info, str):
        info = json.dumps(info, ensure_ascii=False)
    with open(new_file, 'a', encoding='utf-8') as fin:
        fin.write(info + '\n')

def transfer_section_content(sections):
    content = []
    for idx, section in enumerate(sections):
        if "content" in section:
            cur = f"Section {str(idx+1)}: \n{section['content']}"
        elif "section" in section:
            cur = f"Section {str(idx+1)}: \n{section['section']}"
        content.append(cur)
    return "\n---\n".join(content)

def process_line(line):
    question = line['question']
    if "sections_content" in line:
        model_output = line['sections_content']
    else:
        model_output = line['section_content']

    section_index = line['evaluation_section_number']
    idea_error_section_numbers = line['idea_error_section_numbers']
    error_section_numbers = line['error_section_numbers']
    
    all_section_indexs = idea_error_section_numbers + error_section_numbers
    all_section_indexs = list(set(all_section_indexs))
    if "content" in line['sections'][section_index-1]:
        section_content = line['sections'][section_index-1]['content']
    else:
        section_content = line['sections'][section_index-1]['section']
    parsing_success = line.get("parsing_success", 0)
    evaluation_label = line['evaluation_label']
   
    messages = []
    prompt = critic_judge_prompt.replace("{{question}}", question).replace("{{model_output}}", model_output)
    messages.append({"role": "user", "content": prompt})
    line['messages'] = messages
    try:
        critic = line.get("critic", "")
        token_info = {}
        if critic == "":
            output,token_info = call_idealab(messages, call_modelname)
            critic = output 
        print(critic)
        line['critic'] = critic
        line['token_info'] = token_info
            
        if isinstance(critic, str) and critic == "":
            line['parsing_success'] = 0
            line['info'] = "output is None"
            write_to_file(line)
            return 0
        
        info = parse_output(critic, evaluation_label, all_section_indexs)
        line.update(info)
        write_to_file(line)
        return info['parsing_success']
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        write_to_file(line)
        return 0

    
def calculate_accuracies_v2(group):

    total_questions = len(group)
    
    total_actual_errors = group[group['evaluation_label'] == 1].shape[0]
    total_actual_correct = group[group['evaluation_label'] == 0].shape[0]
    
    total_predicted_errors = group[group['judge'] == 1].shape[0]
    total_predicted_correct = group[group['judge'] == 0].shape[0]

    true_positive = group[(group['judge'] == 1) & (group['evaluation_label'] == 1) & (group['correct'] == 1)].shape[0] 
    true_negative = group[(group['judge'] == 0) & (group['evaluation_label'] == 0)].shape[0]
    false_positive = group[(group['judge'] == 1) & (group['correct'] == 0)].shape[0]
    false_negative = group[(group['judge'] == 0) & (group['evaluation_label'] == 1)].shape[0]
    
    accuracy = (true_positive + true_negative) / total_questions if total_questions > 0 else 0
    precision = true_positive / total_predicted_errors if total_predicted_errors > 0 else 0
    recall = true_positive / total_actual_errors if total_actual_errors > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = false_positive / total_actual_correct if total_actual_correct > 0 else 0
    fnr = false_negative / total_actual_errors if total_actual_errors > 0 else 0
    
    precision_macro = group['precision'].mean()
    recall_macro = group['recall'].mean()
    f1_score_macro = group['f1_score'].mean()
    
    
    positive_samples = group[group['evaluation_label'] == 1]
    sum_tp = positive_samples['tp_step'].sum()
    sum_fp = positive_samples['fp_step'].sum()
    sum_fn = positive_samples['fn_step'].sum()
    
    precision_micro = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    recall_micro = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
    
    return pd.Series({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'recall_macro': recall_macro,
        'precision_macro': precision_macro,
        'f1_score_macro': f1_score_macro,
        'recall_micro': recall_micro, 
        'precision_micro': precision_micro,  
        'f1_micro': f1_micro,  
    })

call_modelname = sys.argv[1]
origin = sys.argv[2]
prompt_version = int(sys.argv[3])
vote = int(sys.argv[4])
if len(sys.argv) == 6:
    dataset = sys.argv[5]
else:
    dataset = "human_labeled_v4"


print(f"Model Name: {call_modelname}")
print(f"Origin: {origin}")
print(f"prompt_version: {prompt_version}")
print(f"dataset: {dataset}")


origin_file = "data/Deltabench_v1.jsonl"
new_file = f"evaluation/{dataset}_promptv{str(prompt_version)}_{call_modelname}_vote{str(vote)}.jsonl"


# ==============================
      
if __name__ == "__main__":
    import time
    start_time = time.perf_counter()   
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
    fin =  open(new_file, "w", encoding='utf-8')
    if "deepseek" in call_modelname:
        process_num = 1
    else:
        process_num = 10
        
    with Pool(processes=process_num) as pool:
        sleep(1)
        results = list(tqdm(pool.imap(process_line, data_new), total=len(lines)))       
        correct = np.sum(np.array(results))
        print("success num: ", correct)   
    k = 0 
    all_num = len(data_new)
    all_num = int(all_num*0.95)
    while correct < all_num and k < 3:
        k += 1
        print("fail num: ", correct)
        start_time = time.perf_counter()    
        origin_file = new_file
        with open(origin_file, "r", encoding='utf-8') as fin:
            lines = fin.readlines()
            lines = [json.loads(line) for line in lines]
        new_file = f"{new_file}_{k}.jsonl"
        fin =  open(new_file, "w", encoding='utf-8')
        with Pool(processes=1) as pool:
            results = list(tqdm(pool.imap(process_line, lines), total=len(lines)))       
            correct = np.sum(np.array(results))
            print("success num: ", correct)   
    end_time = time.perf_counter()
    execution_time_ms = (end_time - start_time) / 60
    print(f"time: {execution_time_ms} mins")
    with open(new_file, "r", encoding='utf-8') as fin:
        lines = fin.readlines()
        datas = [json.loads(line) for line in lines]
        df = pd.json_normalize(datas)
        accuracy_df = calculate_accuracies_v2(df)

        overall_row = pd.DataFrame({
            'task_l1': ['Overall'],
            'accuracy': [accuracy_df['accuracy']],
            'precision': [accuracy_df['precision']],
            'recall': [accuracy_df['recall']],
            'f1_score': [accuracy_df['f1_score']],
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