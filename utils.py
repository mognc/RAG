import os
import torch

def load_passages_from_file(input_file_path):
    passages = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
        passages = content.split('\n\n')
    return [passage.strip() for passage in passages if passage.strip()]

def load_embeddings(file_path):
    embeddings = torch.load(file_path)
    return embeddings