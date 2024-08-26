#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
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
from llm_oai import LLMs

from prompt4evaluation import prompt_wo_context_optima, prompt_w_context
from prompt4evaluation import prompt_wo_context_new as prompt_wo_context
from prompt4evaluation import prompt_wo_context_open_new as prompt_wo_context_open

def process(feedback):
    if "```" not in feedback:
        return json.loads(feedback, strict=False)
    elif "```json" in feedback:
        feedback = feedback.split("```json")[1].split("```")[0]
        return json.loads(feedback, strict=False)
    elif "```" in feedback:
        return json.loads(feedback.split("```")[1], strict=False)
    else:
        return None

global_save_path = "your_save_path_here"

def filter_data(data, filter_files):
    if filter_files is None:
        return data
    filter_ids = set([])
    for file in filter_files.split(","):
        if os.path.exists(file) is False:
            print(f"File {file} not found")
            continue
        fdata = srsly.read_json(file)
        for s in fdata:
            if s.get("feedback.json") is not None:
                filter_ids.add(s["id"])
    filter_data = []
    for s in data:
        if s["id"] not in filter_ids:
            filter_data.append(s)
    print(Fore.RED + f"Filtering {len(data)} -> {len(filter_data)} ({len(filter_ids)}) samples" + Style.RESET_ALL)
    return filter_data

def load_data(filename, num_samples, add_optima, add_context, filter_files, is_openq):
    data = srsly.read_json(filename)
    data = filter_data(data, filter_files)
    
    if num_samples is not None:
        data = random.sample(data, min(num_samples, len(data)))
    
    if is_openq:
        print(Fore.RED + f"Template: Prompt w/o optima, w/o context, but openq" + Style.RESET_ALL)
        prompt_format = prompt_wo_context_open
    elif not add_optima and not add_context:
        print(Fore.RED + f"Template: Prompt w/o optima, w/o context" + Style.RESET_ALL)
        prompt_format = prompt_wo_context
    elif add_optima and not add_context:
        print(Fore.RED + f"Template: Prompt with optima, w/o context" + Style.RESET_ALL)
        prompt_format = prompt_wo_context_optima
    elif add_context and not add_optima:
        print(Fore.RED + f"Template: Prompt w/o optima, with context" + Style.RESET_ALL)
        prompt_format = prompt_w_context
    else:
        print(Fore.RED + f"Template: Prompt with optima, with context" + Style.RESET_ALL)
        prompt_format = prompt_wo_context

    prompts = []
    show_count = 0
    for d in tqdm(data):
        # random.shuffle(d["model_outputs"])
        if show_count < 3:
            print([list(mo.keys())[0] for mo in d["model_outputs"]])
            show_count += 1
        candidates = "\n\n".join([f"### Model {str(i)}\n{list(o.values())[0].strip()}" for i, o in enumerate(d['model_outputs'])])
        question = d['conversations'][0]['value']
        if is_openq:
            prompt = prompt_format.format(question=question, candidates=candidates)
        else:
            answer = d['answer']
            prompt = prompt_format.format(question=question, candidates=candidates, answer=answer)
        prompts.append(prompt)
    return data, prompts

def threaded_request(llms_instance, rdata):
    rdata["result"] = llms_instance.request(prompt=rdata["prompt"])
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
        filename = f"{global_save_path}_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        srsly.write_json(filename, results)
        print(f"Progress saved to {filename}")
        exit(0)

    print("Requesting finished")
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time}")
    return results

def main(filename="data/medqa_outputs.json",  # test, val, train-sampling106k
         model_name="gpt-4-1106-preview",
         filter_files=None,
         add_optima=False,
         add_context=False,
         debug=False, # debug for showing samples
         num_samples=None, # number of samples to evaluate
         multi_threads=200, # number of threads for multi-threading
         is_greedy=False,
         is_show_samples=False,
         is_openq=False
        ):
    print(Fore.GREEN + f"File Name: {filename}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Model Name: {model_name}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Add Optima: {add_optima}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Add Context: {add_context}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Debug: {debug}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Num Samples: {num_samples}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Multi Threads: {multi_threads}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Is Greedy: {is_greedy}" + Style.RESET_ALL)

    data, prompts = load_data(filename, num_samples, add_optima, add_context, filter_files, is_openq)
    if is_show_samples:
        for prompt in random.sample(prompts, 3):
            print(prompt)
            print()
    print(Fore.BLUE + f"Number of samples: {len(data)}" + Style.RESET_ALL)
    
    global global_save_path
    global_dir = "/".join(filename.split("/")[:-1]) + "/temp/"
    if os.path.exists(global_dir) is False:
        os.makedirs(global_dir)
    global_save_path = global_dir + filename.split("/")[-1].strip(".json")
    print(Fore.BLUE + f"Global Save Path: {global_save_path}" + Style.RESET_ALL)

    if not debug:
        prompts_dict = [{"prompt": prompt, "id": idx, "did": data[idx]["id"]} for idx, prompt in enumerate(prompts)]
    
        # assert model_name in ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"], f"Unknown model name: {model_name}"
        
        model = {"model": model_name, "request_type": "openai",  "is_greedy": is_greedy}
        print("loading", model_name)
        llm = LLMs(**model)
        results = multi_thread_request(llm, prompts_dict, multi_threads)
        
        for result in results:
            try:
                data[result["id"]]["feedback"] = result["result"]
            except Exception as e:
                raise ValueError(f"Error: {e}")
        
        for d in data:
            if d.get("feedback", None) is not None:
                try:
                    feedback = process(d.get("feedback"))
                except:
                    feedback = None
                finally:
                    d["feedback.json"] = feedback

        temp = "" + ("_optima" if add_optima else "") + ("_context" if add_context else "")
        temp = temp + f"_{num_samples}samples" if num_samples is not None else temp
        output_filename = filename.replace(".json", f"{temp}_{model_name}_feedback_{datetime.now(tz=beijing_tz).strftime('%Y-%m-%d_%H-%M-%S')}.json")
        print(Fore.GREEN + f"Writing to {output_filename}" + Style.RESET_ALL)
        srsly.write_json(output_filename, data)

if __name__ == "__main__":
    Fire(main)