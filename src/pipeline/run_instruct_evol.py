#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import signal
from datetime import datetime
import pytz
# Define Beijing timezone
beijing_tz = pytz.timezone('Asia/Shanghai')
import os
import json
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
from prompt4evlove import complex_prompt, complex_prompt_mc

global_save_path = "your_save_path_here"

def process(feedback):
    if "```" not in feedback:
        return json.loads(feedback, strict=False)
    elif "```json" in feedback:
        feedback = feedback.split("```json")[1].split("```")[0]
        return json.loads(feedback, strict=False)
    elif "```" in feedback:
        return json.loads(feedback.split("```")[1], strict=False)

def threaded_request(llms_instance, rdata):
    try:
        rdata["result"] = llms_instance.request(prompt=rdata["prompt"])
    except Exception as exc:
        print(f'An exception occurred: {exc}')
        rdata["result"] = None
    return rdata

def multi_thread_request(llm_instance, prompts, num_threads=200):
    print(f"Requesting {len(prompts)} prompts with {num_threads} threads")
    start_time = datetime.now()
    results = []
    try:
        # Using ThreadPoolExecutor to manage a pool of threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks to the executor and store future objects
            futures = [executor.submit(threaded_request, llm_instance, prompt) for prompt in prompts]

            # Collect results as they become available
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(prompts), desc="Requesting"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f'An exception occurred: {exc}')
                    results.append(None)
    except KeyboardInterrupt:
        print("Interrupt received, saving progress...")
        filename = f"{global_save_path}_evol_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        srsly.write_json(filename, results)
        print(f"Progress saved to {filename}")
        exit(0)

    print("Requesting finished")
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time}")
    return results

def load_data_and_prompts(filename, num_samples, is_filter, target_key, is_mc):
    data = srsly.read_json(filename)
    data = random.sample(data, num_samples) if num_samples is not None else data
    if is_filter:
        data = filter_cache(filename, data)

    prompts = []
    for sample in data:
        prompt_template = complex_prompt_mc if is_mc else complex_prompt
        prompt = prompt_template.format(question=sample["conversations"][0]["value"] if target_key is None else sample[target_key])
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
                tmp = srsly.read_json(os.path.join(cache_path, file))
                for d in tmp:
                    if d['result'] is not None:
                        ids.add(d["did"] if "did" in d else d["id"])
            except Exception as e:
                print(f"Error: {e}", file)
    filter_data = [d for d in data if d["id"] not in ids]
    print(f"Filtered samples", len(data), "->", len(filter_data), "for", task)
    return filter_data

def extract_base(model_output):
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
         model_name="gpt-4-1106-preview",
         is_filter=False,
         target_key=None,
         output_key="evol",
         is_mc=False
        ):
    print(Fore.GREEN + f"Debug: {debug}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Num Samples: {num_samples}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Multi Threads: {multi_threads}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Is Greedy: {is_greedy}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Model Name: {model_name}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Is Filter: {is_filter}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Target Key: {target_key}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Output Key: {output_key}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Is MC: {is_mc}" + Style.RESET_ALL)
    
    data, prompts = load_data_and_prompts(filename, num_samples, is_filter, target_key, is_mc)
    
    temp_name = f"_{output_key}_{model_name}"
    temp_name = temp_name + "_greedy" if is_greedy else temp_name
    temp_name = temp_name + f"_{num_samples}samples" if num_samples is not None else temp_name
    filename = filename.split(".")[0] + temp_name + "_"
    
    global global_save_path
    global_dir = "/".join(filename.split("/")[:-1]) + "/temp/"
    if os.path.exists(global_dir) is False:
        os.makedirs(global_dir)
    global_save_path = global_dir + filename.split("/")[-1]
    print(Fore.BLUE + f"Global Save Path: {global_save_path}" + Style.RESET_ALL)
    print(Fore.BLUE + f"Number Samples: {len(data)}" + Style.RESET_ALL)

    if not debug:
        prompts_dict = [{"prompt": prompt, "id": idx, "did": data[idx]['id']} for idx, prompt in enumerate(prompts)]
    
        # assert model_name in ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"], f"Unknown model name: {model_name}"
        
        model = {"model": model_name, "request_type": "openai",  "is_greedy": is_greedy}
        print("loading", model_name)
        llm = LLMs(**model)
        results = multi_thread_request(llm, prompts_dict, multi_threads)

        for result in results:
            try:
                data[result["id"]]["result"] = result["result"]
                # data[result["id"]][output_key] = process(result["result"])["question"]
                # data[result["id"]]["answer"] = process(result["result"])["answer"]
                # data[result["id"]]["conversations"][1]["value"] = f"The answer is {process(result['result'])['answer']}."
            except Exception as e:
                pass

        filename += datetime.now(tz=beijing_tz).strftime('%Y-%m-%d_%H-%M-%S') + ".json"
        print(Fore.GREEN + f"Writing to {filename}" + Style.RESET_ALL)
        srsly.write_json(filename, data)

if __name__ == "__main__":
    Fire(main)