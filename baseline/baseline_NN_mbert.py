from pprint import pprint
import pandas as pd
from load_data import load_conll_data

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AdamW
from torch.nn import CrossEntropyLoss

from help import get_lemmas_from_stanza_list

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

# read the json of the preprocessed data
with open('../data/output/preprocessing_outputs/sample_preprocessed.json') as f:
    sample = json.load(f)

long_data = []
for obj in sample:
    # now create a new object for each preprocessing mode output token and append it to the long_data list
    # we iterate over the processed token objects, with the iterating number being i
    # the ith token should correspond with the ith label
    # TODO: should we use the full original query or a concatenated version of its lemmas?
    query = obj.get("model_input")
    for sentence in obj["model_output_text_processed"]:
        for token in sentence:
            lemma = token.get("lemma")
            upos = token.get("upos")
            xpos = token.get("xpos")
            label = token.get("hallucination")
            long_data.append({
                "query": query,
                "lemma": lemma,
                "upos": upos,
                "xpos": xpos,
                "label": int(label) if label is not None else 0
            })

# [CLS] query [SEP] a single token from the answer [SEP] UPOS: the upos of the token, XPOS: the xpos of the token [SEP]
features = [f"[CLS] {obj.get('query')} [SEP] {obj.get('lemma')} [SEP] UPOS: {obj.get('upos')} [SEP] {obj.get('xpos')}"
            for obj in long_data]
labels = [obj.get('label') for obj in long_data]

if len(features) != len(labels):
    raise Exception("The number of features and labels do not match!")
print("Data is prepared!")

# Define constants
MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 128  # Max token length for mBERT

print("Loading tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
print("Tokenizer and model loaded!")


# create a class for the dataset
class HallucinationDataset(Dataset):
    def __init__(self, features, labels, tokenizer, max_length):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        # Tokenize input text
        encoded = self.tokenizer(
            feature,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# Create dataset and dataloader
dataset = HallucinationDataset(features, labels, tokenizer, max_length=MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
print("Dataset and dataloader created!")

# Training setup

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss()
print("Optimizer and loss function defined!")
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model.to(device)
print("Model moved to device!")

# Step 5: Training loop
model.train()
print("Training started!")
for epoch in range(3):  # Number of epochs
    total_loss = 0
    for batch in dataloader:
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
print("Training complete!")
# Step 6: Save the model
model.save_pretrained("mbert_token_classifier")
tokenizer.save_pretrained("mbert_token_classifier")
print("Model and tokenizer saved!")