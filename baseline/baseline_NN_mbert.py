import json
import torch
from transformers import AdamW
from torch.nn import CrossEntropyLoss

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from baseline_utils import get_data_for_training

import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

# To Niko: exported as a function to baseline_utils for clearer struct, it was the same for both models
features, labels = get_data_for_training()
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