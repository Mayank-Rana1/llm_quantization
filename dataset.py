from datasets import load_dataset

def get_dataset():
    return load_dataset("roneneldan/TinyStories")

dataset = get_dataset()
print(dataset)
import torch 
print(torch.cuda.is_available)