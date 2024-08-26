#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import random
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain.chat_models import ChatOpenAI

def request_oai(name, description, task = "domains", number=20, model="gpt-4-1106-preview", parameters={"top_p": 0.7, "temperature": 0.9}):
    chat = ChatOpenAI(model=model)
    chat.model_kwargs = parameters
    assert task in ["domains", "instructions"], "task should be one of ['domains', 'instructions']"

    prompt4domains = f"""{name}: {description}
As an expert in the field of {name}, I need you to do the following:
1. List {number} subfields within the realm of {name} research.
2. Ensure that these subfields represent distinct areas of {name} without any overlap.
3. Provide a brief description for each subfield, highlighting its main research focus and characteristics.
4. Aim for this list to comprehensively reflect the diversity and breadth of the biomedical field.
5. Present this list in an array of dictionaries format, with each dictionary containing two keys: 'name' (the name of the subfield) and 'description' (a brief description of the subfield).

Example output format:
[
  {{"name": "Gene Editing", "description": "Gene editing involves altering the genetic material of organisms to study gene functions or treat genetic diseases."}},
  {{"name": "Neuroscience", "description": "Neuroscience focuses on the study of the structure, function, and diseases of the nervous system."}},
  // ... 18 more subfields
]"""

    prompt4instructions = f"""{name}: {description}
As an expert in the field of {name}, please devise {number} {name}-related questions or instructions, formatted as an array of dictionaries, each with two keys: 'instruction' and 'context'. Follow these guidelines:
1. **Verb Diversity**: Incorporate a broad spectrum of verbs to diversify and enrich the instructions set.
2. **Language Style Variability**: Blend both interrogative and imperative sentence structures to enhance the dynamism of instructions.
3. **Range of Task Types**: Ensure the tasks span a variety of categories such as explanations, analyses, comparisons, and more. 1. **Difficulty levels should vary from elementary concepts to complex scientific inquiries and extend to addressing novel, challenging scenarios.
4. **Exclusivity to Text-Based Tasks**: Frame all instructions in a text-only format. Refrain from incorporating tasks that require physical execution or laboratory experimentation.
5. **Conciseness and Precision**: Articulate each instruction in English with utmost precision, limiting it to 1 or 2 sentences for clarity and brevity.
6. **Background Information Accuracy**: For tasks necessitating supplementary context, provide succinct yet comprehensive descriptions (restricted to 100 words). For basic queries, simply state "None" in the context section.
7. **JSON Format Adherence**: Format the output as an array of dictionaries. Each dictionary should have two keys: 'instruction' for the task description and 'context' for the relevant background information.

Example output format:
[
  {{"instruction": "Explain the structure of liposomes and their role in drug delivery.", "context": "Liposomes are nanoscale carriers used in drug delivery, where their structure and function significantly impact efficiency."}},
  {{"instruction": "List three common cardiovascular diseases.", "context": "None"}},
  // ... 18 more instructions
]"""

    batch_messages = [[
        HumanMessage(content=prompt4domains if task == "domains" else prompt4instructions),
    ]]
    results = chat.generate(batch_messages)

    for result in results.generations:
        print(result[0].text)
        text = result[0].text
        if "```json" in text:
            text = text.split("```json")[1].replace("```", "").strip()
        elif "```" in text:
            text = text.split("```")[1].strip()
        
        json.dump(json.loads(text), open(f"gpt4/{task}/{name}.json", "w"), ensure_ascii=False, indent=4)
        # srsly.write_json(f"gpt4/{task}/{name}.json", json.loads(result[0].text))
    return json.loads(text)

if __name__ == "__main__":
    # domains = [
    #     {"name": "Genomics", "description": "Genomics involves the comprehensive study of genomes, the complete set of DNA within an organism, to understand genetic variation and function in health and disease."},
    #     {"name": "Proteomics", "description": "Proteomics is the large-scale study of proteomes, the entire complement of proteins produced by an organism, to understand protein structure, function, and interactions."},
    #     {"name": "Immunology", "description": "Immunology is the study of the immune system, its components, how it fights disease, and how it can be harnessed for therapies."},
    #     {"name": "Pharmacology", "description": "Pharmacology examines the interactions between drugs and biological systems, aiming to discover and develop new medications."},
    #     {"name": "Bioinformatics", "description": "Bioinformatics combines biology, computer science, and information technology to analyze and interpret biological data, such as genetic sequences."},
    #     {"name": "Regenerative Medicine", "description": "Regenerative medicine focuses on repairing or replacing damaged cells, tissues, and organs through techniques like stem cell therapy and tissue engineering."},
    #     {"name": "Medical Imaging", "description": "Medical imaging uses techniques such as MRI, CT scans, and X-rays to visualize the interior of the body for diagnosis and treatment planning."},
    #     {"name": "Biomechanics", "description": "Biomechanics applies principles of mechanics to understand the movement and structure of biological systems, with applications in prosthetics and injury prevention."},
    #     {"name": "Toxicology", "description": "Toxicology studies the harmful effects of chemical substances on living organisms and the environment, with the goal of improving safety and managing risks."},
    #     {"name": "Epidemiology", "description": "Epidemiology is the study of the distribution, patterns, and determinants of health and disease conditions in defined populations."},
    #     {"name": "Cancer Biology", "description": "Cancer biology investigates the molecular and cellular basis of cancer to develop better diagnostics, treatments, and preventive strategies."},
    #     {"name": "Molecular Biology", "description": "Molecular biology explores the molecular underpinnings of biological processes, including gene expression, regulation, and protein function."},
    #     {"name": "Nutritional Science", "description": "Nutritional science examines the impact of food and nutrients on health, metabolism, and disease prevention."},
    #     {"name": "Biomedical Engineering", "description": "Biomedical engineering combines engineering principles with medical and biological sciences to design and create equipment, devices, computer systems, and software used in healthcare."},
    #     {"name": "Neurobiology", "description": "Neurobiology studies the biology of the nervous system, including its structure, function, development, and pathology."},
    #     {"name": "Virology", "description": "Virology is the study of viruses and virus-like agents, their structure, function, and their impact on living organisms."},
    #     {"name": "Cardiovascular Biology", "description": "Cardiovascular biology focuses on the study of the heart and blood vessels, their function, and related diseases."},
    #     {"name": "Developmental Biology", "description": "Developmental biology investigates the process by which organisms grow and develop, from fertilization to maturity."},
    #     {"name": "Biogerontology", "description": "Biogerontology studies the biological processes of aging and the associated changes in organisms over time."},
    #     {"name": "Microbiome Research", "description": "Microbiome research explores the complex communities of microorganisms that inhabit various environments within the human body and their relationship to health and disease."}
    # ]
    
    # for domain in domains:
    #     print(1, domain)
    #     try:
    #         results = request_oai(domain["name"], domain["description"], task="domains", number=20, model="gpt-4-1106-preview", parameters={"top_p": 0.7, "temperature": 0.9})
    #     except Exception as e:
    #         print(e)
    #         results = None
    #         continue
    #     domain["subdomains"] = results
    
    random.seed(42)
    data = json.loads(open("data/biomed_domains.json").read())
    for domain in data:
        try:
            results = request_oai(domain["name"], domain["description"], task="instructions", number=20, model="gpt-4-1106-preview", parameters={"top_p": 0.7, "temperature": 0.9})
        except Exception as e:
            print(e)
            results = None
            continue
        finally:
            domain["instructions"] = results
    
    json.dump(data, open("outputs/biomed_domains.json", "w"), ensure_ascii=False, indent=4)