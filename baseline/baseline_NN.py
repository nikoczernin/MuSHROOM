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

from help import get_lemmas_from_stanza_list

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


# read the json of the preprocessed data
with open('../data/output/preprocessing_outputs/sample_preprocessed.json') as f:
    sample = json.load(f)


# TODO: we now have a list of objects. Now Create a long dataset with 1 reponse token per row
# iterate over the data.model_output_text_lemmas and get a object with the lemma, upos, xpos and a binary label
long_data = []

# TODO: also before that, check whether the lemmas list is still corresponding with the processed text hard labels!
for obj in sample:
    # complete the hard labels
    # they are currently a list of lists of spans, we need a list of indices instead
    obj["labels_complete"] = [list(range(left, right+1)) for left, right in obj["hard_labels"]]

    # Stanza preprocessing saved punctuation as its own tokens
    # the hard labelling we did ignored punctuation, so we need to remove them at this point
    obj["model_output_text_processed"] = [token for token in obj["model_output_text_processed"] if token["upos"] != "PUNCT"]

    # now create a new object for each preprocessing mode output token and append it to the long_data list
    # we iterate over the processed token objects, with the iterating number being i
    # the ith token should correspond with the ith label
    for i, token in enumerate(obj["model_output_text_processed"]):
        long_data.append({
            "lemma": token["lemma"],
            "upos": token["upos"],
            "xpos": token["xpos"],
            "label": 1 if i in obj["labels_complete"] else 0
        })




def annotate_response_tokens(response, hallucinations):
    """
    Annotate response tokens with hallucination labels.
    Args:
        response: List of token dictionaries (response).
        hallucinations: List of token index ranges indicating hallucinated spans.
    Returns:
        List of dictionaries containing token features and labels.
    """
    # Flatten hallucination ranges into a set for quick lookup
    hallucination_ids = set()
    for start, end in hallucinations:
        hallucination_ids.update(range(start + 1, end + 2))

    # Annotate tokens
    annotated_data = []
    for token in response:
        token_id = token['id']
        if isinstance(token_id, list):
            print(token_id)
            continue # this excludes where id is made up of multiple tokens (for example: it's) since the next tokens include it anyway
        label = 1 if token_id in hallucination_ids else 0
        annotated_data.append({
            'features': {
                'lemma': token['lemma'],
                'upos': token['upos'],
                'xpos': token['xpos']
                # Do we need more features?
            },
            'label': label
        })
    return annotated_data

responses = [token for sentence in sandor['model_output_text_processed'] for token in sentence]
annot = annotate_response_tokens(responses, sandor['hard_labels'])


########################
# MODEL TEST


# Initialize label encoders for each feature
lemma_encoder = LabelEncoder()
upos_encoder = LabelEncoder()
xpos_encoder = LabelEncoder()

# Collect all the lemmas, upos, and xpos for encoding
lemmas = [item['features']['lemma'] for item in annot]
uposes = [item['features']['upos'] for item in annot]
xposes = [item['features']['xpos'] for item in annot]

# Fit the encoders
lemma_encoder.fit(lemmas)
upos_encoder.fit(uposes)
xpos_encoder.fit(xposes)

# Prepare the data for training
X = []
y = []

for item in annot:
    lemma_encoded = lemma_encoder.transform([item['features']['lemma']])[0]
    upos_encoded = upos_encoder.transform([item['features']['upos']])[0]
    xpos_encoded = xpos_encoder.transform([item['features']['xpos']])[0]

    # Create a feature vector (lemmas, upos, xpos) for each token
    feature_vector = [lemma_encoded, upos_encoded, xpos_encoded]
    X.append(feature_vector)
    y.append(item['label'])

# Convert to numpy arrays for training
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

raise hell


# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)  # for classification, use torch.long for labels

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


class HallucinationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(HallucinationModel, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # Input to first hidden layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # First hidden to second hidden layer
        self.fc3 = nn.Linear(hidden_dim2, output_dim)  # Second hidden to output layer

        # Define activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass input through layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define input, hidden, and output dimensions
input_dim = 3  # 3 features: lemma, upos, xpos
hidden_dim1 = 64
hidden_dim2 = 32
output_dim = 2  # Binary classification: 0 or 1

# Initialize the model
model = HallucinationModel(input_dim, hidden_dim1, hidden_dim2, output_dim)

# Print model summary
print(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Since it's a classification task
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    model.train()

    # Loop through batches (for simplicity, no batching in this example)
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print loss for every epoch
    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    # Get predictions on validation set
    val_outputs = model(X_val)
    _, predicted = torch.max(val_outputs, 1)  # Get the index with the highest score

    # Calculate accuracy
    correct = (predicted == y_val).sum().item()
    accuracy = correct / y_val.size(0)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

print("nn")