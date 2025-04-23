# inbuilt
import json

# 3rd parties
from torch.utils.data import Dataset

class TripletDataset(Dataset):

    def __init__(self, tokenizer, filepath:str):
        self.dataset = []

        
        with open(filepath, "r") as f:
            for line in f:
                self.dataset.append(json.loads(line))            
    
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]

        # Tokenize anchor, positive, and negative examples
        anchor = self.tokenizer(row["anchor"], padding="max_length", truncation=True, return_tensors="pt", max_length=128)
        positive = self.tokenizer(row["positive"], padding="max_length", truncation=True, return_tensors="pt", max_length=128)
        negative = self.tokenizer(row["negative"], padding="max_length", truncation=True, return_tensors="pt", max_length=128)

        # Flatten the tensors (remove extra batch dimension) and create a dictionary with the right keys
        # These keys are what the model expects for training
        return {
            "input_ids": anchor["input_ids"].squeeze(0),  # For anchor
            "attention_mask": anchor["attention_mask"].squeeze(0),  # For anchor
            "positive_input_ids": positive["input_ids"].squeeze(0),  # For positive sample
            "positive_attention_mask": positive["attention_mask"].squeeze(0),  # For positive sample
            "negative_input_ids": negative["input_ids"].squeeze(0),  # For negative sample
            "negative_attention_mask": negative["attention_mask"].squeeze(0),  # For negative sample            
        }