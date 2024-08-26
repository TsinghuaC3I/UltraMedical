#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import faiss
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import random
from tqdm import tqdm
from loguru import logger
from types import ModuleType
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModel

import srsly

def to_multi_gpu(index_flat):
    num_gpus = faiss.get_num_gpus()

    gpu_resources = []
    for i in range(num_gpus):
        res = faiss.StandardGpuResources()  # Initialize resources for each GPU
        gpu_resources.append(res)

    dimension = 512  # Example dimension size of your vectors
    # index = faiss.IndexFlatL2(dimension)  # Or any other index
    multi_gpu_index = faiss.IndexShards(dimension)

    for i in range(num_gpus):
        gpu_index = faiss.index_cpu_to_gpu(gpu_resources[i], i, index_flat)
        multi_gpu_index.add_shard(gpu_index)

    return multi_gpu_index

def to_gpu(index_flat):
    gpu_resources = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index_flat)
    return gpu_index

def get_sentence_from_sample(sample):
    sentence = sample["conversations"][0]["value"]
    if "pubmedqa" in sample["id"]:
        sentence = "\n".join(sentence.split("\n")[:-3])
    else:
        sentence = "\n".join(sentence.split("\n")[:-4])
    return sentence

def main():
    model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-small-en", # switch to en/zh for English or Chinese
    trust_remote_code=True
    )
    # control your input sequence length up to 8192
    model.max_seq_length = 1024

    print("Loading all data")
    datasets = srsly.read_json("data/ultramecial.json")
    
    print("Loading all test data")
    testsets = srsly.read_json("data_sampling/outputs/reformat-data-test.json")
    test_sentences = []
    test_length = []
    for sample in testsets:
        sentence = get_sentence_from_sample(sample)
        test_sentences.append(sentence)
        test_length.append(len(sentence))
    
    print("Loading all sentence")
    for sentence in random.sample(test_sentences, 5):
        print(sentence)
        print()
    print("encoding test data")
    test_embs = model.encode(test_sentences)
    
    index_cpu = faiss.IndexFlatL2(512)
    index = to_multi_gpu(index_cpu)
    index.add(test_embs)
    
    remove = set([])
    batch = []
    dataset_size = len(datasets)
    for dataset_idx in tqdm(range(dataset_size)):
        item = datasets[dataset_idx]
        batch.append({
            "text": get_sentence_from_sample(item),
            "id": item["id"]
        })
        if len(batch) == 1024 or dataset_idx == dataset_size - 1:
            embeddings = model.encode([batch_item["text"] for batch_item in batch])
            distances, indices = index.search(embeddings, k=1)
            for idx in range(len(batch)):
                if not len(distances[idx]):
                    continue
                distance = distances[idx][0]
                found_index = indices[idx][0]
                # Not sure what's actually best here, but I'm going with:
                # cos sim > 0.05 or > 20% diff in length = not contamination.
                length_delta = abs(
                    (len(batch[idx]["text"]) - test_length[found_index])
                    / (len(batch[idx]["text"]) or 1)
                )
                if distance <= 0.05 and length_delta <= 0.20:
                    logger.warning(f"Likely contamination: {batch[idx]['id']}")
                    remove.add(batch[idx]["id"])
            batch = []
    srsly.write_json("/root/kyzhang/llms/UltraMedical/data/ultramecial_remove_ids.json", list(remove))
    
if __name__ == "__main__":
    main()