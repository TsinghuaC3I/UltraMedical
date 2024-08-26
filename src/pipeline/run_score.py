#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import signal
from datetime import datetime
import pytz
# Define Beijing timezone
beijing_tz = pytz.timezone('Asia/Shanghai')
import os
import srsly
import random
random.seed(42)
from fire import Fire
from colorama import Fore, Style
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np

from llm_oai import LLMs

direct_score_on_question_prompt = """Please evaluate the following question and rate its difficulty and complexity on a scale from 1 to 10, with 1 being the least difficult/complex and 10 being the most difficult/complex. Consider factors such as the breadth and depth of knowledge required, the number of concepts involved, the level of technical terminology, and the presence of quantitative or analytical components.

In addition to the numerical score, provide a brief justification (1-2 sentences) explaining your rationale for the assigned score. This will help us better understand the reasoning behind your evaluation.

## Question
{question}

## Evaluation
Justification:
Score: [1-10]"""


global_save_path = "your_save_path_here"
global_data_list = []
global_save_interval = 10000

def save_progress():
    filename = f"{global_save_path}_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    srsly.write_json(filename, global_data_list)
    print(f"Progress saved to {filename}")

def signal_handler(sig, frame):
    print("Interrupt received, saving progress...")
    save_progress()
    exit(0)

def threaded_request(llms_instance, rdata):
    try:
        rdata["result"] = llms_instance.request(prompt=rdata["prompt"])
        if rdata["result"] is not None:
            global_data_list.append(rdata)
        if len(global_data_list) % global_save_interval == 0:
            save_progress()
    except Exception as exc:
        print(f'An exception occurred: {exc}')
    return rdata

def multi_thread_request(llm_instance, prompts, num_threads=200):
    
    print(f"Requesting {len(prompts)} prompts with {num_threads} threads")
    start_time = datetime.now()
    # Using ThreadPoolExecutor to manage a pool of threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
       # Submit tasks to the executor and store future objects
        futures = [executor.submit(threaded_request, llm_instance, prompt) for prompt in prompts]

        # Collect results as they become available
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(prompts), desc="Requesting"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'An exception occurred: {exc}')
                results.append(None)
    print("Requesting finished")
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time}")
    return results

def load_data_and_prompts(filename, num_samples, is_filter):
    data = srsly.read_json(filename)
    data = random.sample(data, num_samples) if num_samples is not None else data
    if is_filter:
        data = filter_cache(filename, data)

    prompts = []
    for sample in data:
        prompt = direct_score_on_question_prompt.format(question=sample["conversations"][0]["value"])
        prompts.append(prompt)
    
    print("Samples:", len(prompts))
    for prompt in random.sample(prompts, 2):
        print("=============== Prompt ===============\n", Fore.GREEN + prompt + Style.RESET_ALL)
        print()
    return data, prompts

def filter_cache(filename, data):
    task = filename.split("/")[-1].replace(".json", "")
    ids = set([])
    cache_path = "/".join(filename.split("/")[:-1]) + "/temp"
    for file in tqdm(os.listdir(cache_path), desc=f"Filtering {task}"):
        if task in file:
            try:
                tmp = srsly.read_json(file)
                for d in tmp:
                    if d['result'] is not None:
                        ids.add(d["did"])
            except Exception as e:
                print(f"Error: {e}", file)
    filter_data = [d for d in data if d["id"] not in ids]
    print(f"Filtered samples", len(data), "->", len(filter_data), "for", task)
    return filter_data

def extract_base(model_output):
    if model_output is None:
        return None
    model_output = model_output.strip()

    # pattern = r"\d[0]?"
    regex = r'\b(10|[1-9])\b'
    matches = re.findall(regex, model_output)
    if len(matches) > 0:
        option = matches[0]
    else:
        option = None
    
    return option

def main(filename=None,  # test, val, train-sampling106k
         debug=False, # debug for showing samples
         num_samples=None, # number of samples to evaluate
         multi_threads=200, # number of threads for multi-threading
         is_greedy=False,
         model_name="gpt-3.5-turbo-1106",
         is_filter=False,
        ):
    """
    python run_score.py --filename "data/medqa.json" --model_name "gpt-4-1106-preview"
    """
    print(Fore.GREEN + f"Debug: {debug}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Num Samples: {num_samples}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Multi Threads: {multi_threads}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Is Greedy: {is_greedy}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Model Name: {model_name}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Is Filter: {is_filter}" + Style.RESET_ALL)
    
    data, prompts = load_data_and_prompts(filename, num_samples, is_filter)
    
    temp_name = f"_{model_name}"
    temp_name = temp_name + "_greedy" if is_greedy else temp_name
    temp_name = temp_name + f"_{num_samples}samples" if num_samples is not None else temp_name
    dir_path = "/".join(filename.split("/")[:-1]) + "/temp/"
    filename = filename.split("/")[-1].split(".")[0] + temp_name + "_"
    global global_save_path
    global_save_path = dir_path + filename
    
    if not debug:
        prompts_dict = [{"prompt": prompt, "id": idx, "did": data[idx]['id']} for idx, prompt in enumerate(prompts)]
    
        # assert model_name in ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"], f"Unknown model name: {model_name}"
        
        model = {"model": model_name, "request_type": "openai",  "is_greedy": is_greedy}
        print("loading", model_name)
        llm = LLMs(**model)
        results = multi_thread_request(llm, prompts_dict, multi_threads)
        save_progress()

        for result in results:
            try:
                data[result["id"]]["score"] = result["result"]
            except Exception as e:
                raise ValueError(f"Error: {e}")

        filename = dir_path + filename + datetime.now(tz=beijing_tz).strftime('%Y-%m-%d_%H-%M-%S') + ".json"
        print(Fore.GREEN + f"Writing to {filename}" + Style.RESET_ALL)
        srsly.write_json(filename, data)

        count = []
        for sample in data:
            model_output = sample.get("score", None)
            if model_output is not None:
                option = extract_base(model_output)
            else:
                option = None
            try:
                count.append(1 if 1 <= int(option) <= 10 else 0)
            except:
                count.append(0)
        print(f"Accuracy: {round(np.mean(count), 2)}")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    Fire(main)