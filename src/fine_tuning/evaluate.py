# inbuilt
import os
import json

# 3rd parties
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

# local import
from model.mean_pooling import mean_pooling

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DATASET_NAME = 'gen_z_slang'
EVALUATE_DATASET_PATH = os.path.join("data", "fine_tuning", f"{DATASET_NAME}_evaluation.jsonl")
FINE_TUNED_MODEL_PATH = os.path.join("data", "model", f"{BASE_MODEL_NAME}-ft-{DATASET_NAME}")

EVALUATE_RESULT_PATH = os.path.join("data", "fine_tuning", "output", f"{DATASET_NAME}_evaluation_result.csv")


def calculate_squared_educlidean_distance(text1: str, text2: str, model, tokenizer):
    """
    squared Euclidean distance, to fit faiss usage
    https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances

    """

    def get_embedding(tokenizer, text):
        encoded_input = tokenizer(
            text, padding=True, truncation=True, return_tensors="pt", max_length=512)

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        with torch.no_grad():
            output = model(**encoded_input)
        embeddings = mean_pooling(output, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    text1_emb = get_embedding(tokenizer, text1)
    text2_emb = get_embedding(tokenizer, text2)

    # squared Euclidean distance, to fit faiss usage
    # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
    return F.pairwise_distance(text1_emb, text2_emb, p=2).pow(2)


def read_evaluate_dataset():
        
    data = []

    with open(EVALUATE_DATASET_PATH, "r") as f:
        for line in f:
            row = json.loads(line)
            data.append({'input':row['anchor'],
                         'meaning':row['positive'],
                         'correct': 1
                         })
            data.append({'input':row['anchor'],
                         'meaning':row['negative'],
                         'correct': 0
                         })            

    return pd.DataFrame(data)


def evaluate():

    df = read_evaluate_dataset()
    print(f"loaded ({df.shape}) rows for evaluation")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # calculate distance by base model
    print("calculate distance by base model")
    base_model = AutoModel.from_pretrained(BASE_MODEL_NAME)        
    df['base_distance'] = df.apply(lambda x: calculate_squared_educlidean_distance(
        x['input'], x['meaning'], base_model, tokenizer).item(), axis=1)


    # calculate distance by fine-tuned model
    print("calculate distance by fine-tuned model")
    lora_model = AutoModel.from_pretrained(FINE_TUNED_MODEL_PATH)
    df['ft_distance'] = df.apply(lambda x: calculate_squared_educlidean_distance(
        x['input'], x['meaning'], lora_model, tokenizer).item(), axis=1)


    df['ft_distance'] = df['ft_distance'].astype(float).round(6)
    df['base_distance'] = df['base_distance'].astype(float).round(6)

    df['dist_change'] = df.apply(lambda row: np.round(
        row['ft_distance'] - row['base_distance'], 6), axis=1)
    df['dist_change_pct'] = df.apply(lambda x: (
        x['dist_change'] / x['base_distance']) if x['base_distance'] != 0 else 0, axis=1)
    
    return df

def main():

    df = evaluate()

    # display result in terminal
    samples = {
        'correct': df[(df['correct'] == 1)],    
        'wrong': df[(df['correct'] == 0)],
    }

    for key, df_sample in samples.items():
        print(f"{key} sample size of {key} samples: {len(df_sample):,}")
        print(
            f"{key} sample avg distance before fine tune: {df_sample['base_distance'].mean():.4f}")
        print(
            f"{key} sample avg distance after fine tune: {df_sample['ft_distance'].mean():.4f}")
        print(
            f"{key} sample distance changes: {df_sample['dist_change'].mean():.4f}")
        print(
            f"{key} sample distance changes in pct: {df_sample['dist_change_pct'].mean():.2%}")


    # export result to csv
    df.to_csv(EVALUATE_RESULT_PATH, index=False)

main()