#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import pytz
from tqdm import tqdm
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
from llm_oai import LLMs

# 1. Evaluate the examination significance of the provided paragraph.
# 2. If the paragraph is deemed significant, generate a synthetic multi-choice question based on the paragraph's content and the provided examples.
# 3. The generated question should test the reader's understanding of the key concepts presented in the paragraph.
# 4. Provide the output in the specified JSON format.

prompt4qg_from_textbooks = """## Paragraph from the medical textbook
{paragraph}

## Example multi-choice questions
### Example 1
Question: {example1}
Answer: {answer1}

### Example 2
Question: {example2}
Answer: {answer2}

### Example 3
Question: {example3}
Answer: {answer3}

## Instructions
1. Evaluate the examination significance of the provided paragraph.
2. Assess whether the paragraph contains sufficient knowledge to evaluate a powerful AI like GPT-4. Consider factors such as:
   - Depth and breadth of the medical concepts covered
   - Specificity and technicality of the information provided
   - Potential for testing higher-order thinking skills
3. If the paragraph is deemed significant and contains enough knowledge to evaluate GPT-4, generate a synthetic multi-choice question based on the paragraph's content and the provided examples. Ensure that the generated question has a single, unambiguous correct answer among the provided choices.
4. If the paragraph is not significant or lacks sufficient knowledge for AI evaluation, set the value of "generated_question" to an empty object ({{}}).
5. Provide the output in the specified JSON format.

## Output Format (JSON)
{{
  "examination_significance": boolean,
  "sufficient_knowledge_for_ai_evaluation": boolean,
  "generated_question": {{
    "question": string,
    "answer_choices": [
      {{
        "choice": string,
        "correct": boolean
      }},
      {{
        "choice": string,
        "correct": boolean
      }},
      {{
        "choice": string,
        "correct": boolean
      }},
      {{
        "choice": string,
        "correct": boolean
      }}
    ]
  }}
}}"""

# "explanation": string

def threaded_request(llms_instance, rdata):
    rdata["result"] = llms_instance.request(prompt=rdata["prompt"])
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

def filter_paragraph(raw_data, filter_file):
    print("Loading filter file...", filter_file)
    filters = srsly.read_json(filter_file)
    ids = set([d['id'].split(',')[-1] for d in filters])
    data = []
    for sample in raw_data:
        if sample['id'] not in ids:
            data.append(sample)
    print("Filter:", len(raw_data), "-", len(ids), "=", len(data))
    return data

def main(model_name="gpt-4-1106-preview",
         filter_file=None,
         debug=False, # debug for showing samples
         num_samples=None, # number of samples to evaluate
         multi_threads=200, # number of threads for multi-threading
         target_task="medqa",
         paragraphs_path="data/paragraphs.jsonl",
         medqa_data_path="data/medqa.json",
         output_dir="outputs"
        ):
    print(Fore.GREEN + f"Model Name: {model_name}" + Style.RESET_ALL)
    # print(Fore.GREEN + f"Split: {split}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Debug: {debug}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Num Samples: {num_samples}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Multi Threads: {multi_threads}" + Style.RESET_ALL)
    
    print("Loading textbooks...")
    paragraphs = srsly.read_json(paragraphs_path)
    paragraphs = filter_paragraph(paragraphs, filter_file)
    if num_samples is not None:
        paragraphs = random.sample(paragraphs, num_samples)
    print("Paragraphs:", len(paragraphs))
    print("Loading medqa...")
    medqa_data = srsly.read_json(medqa_data_path)
    medqa_questions = [[sample["conversations"][0]["value"], sample["answer"]] for sample in medqa_data]
    triple_questions = [random.sample(medqa_questions, 3) for _ in range(len(paragraphs))]
    prompts_dict = [{"prompt": prompt4qg_from_textbooks.format(paragraph=paragraph, example1=triple_questions[idx][0][0], answer1=triple_questions[idx][0][1], example2=triple_questions[idx][1][0], answer2=triple_questions[idx][1][1], example3=triple_questions[idx][2][0], answer3=triple_questions[idx][2][1]), "id": idx} for idx, paragraph in enumerate(paragraphs)]

    print("Samples:", len(prompts_dict))
    for prompt in random.sample(prompts_dict, 1):
        print("Prompt: \n", Fore.GREEN + prompt['prompt'] + Style.RESET_ALL)
        print()

    if not debug:
        # assert model_name in ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"], f"Unknown model name: {model_name}"
        
        model = {"model": model_name, "request_type": "openai"}
        print("loading", model_name)
        llm = LLMs(**model)
        results = multi_thread_request(llm, prompts_dict, multi_threads)
        
        for result in results:
            try:
                paragraphs[result["id"]]["model_output"] = result["result"]
            except Exception as e:
                raise ValueError(f"Error: {e}")

        if os.path.exists(f"{output_dir}/{target_task}") is False:
            os.makedirs(f"{output_dir}/{target_task}")
        temp_name = ""
        if num_samples is not None:
            temp_name += "" + str(num_samples) + "samples_"
        output_path = f"{output_dir}/{target_task}/{temp_name}{model_name}_{datetime.now(tz=beijing_tz).strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"
        print(Fore.GREEN + f"Writing to {output_path}" + Style.RESET_ALL)
        srsly.write_json(output_path, paragraphs)

if __name__ == "__main__":
    Fire(main)