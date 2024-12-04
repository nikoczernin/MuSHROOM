from pprint import pprint
import pandas as pd
from load_data import load_conll_data

import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import os


print("hi")

# # read sample data
# sample = pd.read_json('../data/sample/sample_set.v1.json', lines=True)
# # print(sample.iloc[2])


# load the preprocessed sample data from JSON
with open('../data/output/preprocessing_outputs/sample_preprocessed.json') as f:
    samplep = json.load(f)

# pprint(samplep[2])
# df = pandas.read_json([samplep])


def get_lemmas_from_stanza_list(stanza_list, split_ngrams=True, skip_punct=False):
    """
    Extracts lemmas from a Stanza-processed list of sentences, with optional handling of n-grams and punctuation.
    Parameters:
    - stanza_list (list): A list of sentences, where each sentence is a list of word dictionaries.
                          Each word dictionary includes attributes like 'id', 'lemma', 'text', and 'upos'.
                          Example: [{'id': 1, 'lemma': 'example', 'text': 'example', 'upos': 'NOUN'}, ...].
    - split_ngrams (bool): Determines how to handle n-grams:
                           - If True: Keeps only the split components of n-grams (e.g., "don't" becomes ["do", "n't"]).
                           - If False: Skips split components and retains only the joint n-gram (e.g., "don't").
    - skip_punct (bool): If True, skips tokens labeled as punctuation (e.g., ".", ",", "!", etc.).
    Returns:
    - list: A list of lemmas extracted from the input sentences, optionally handling n-grams and skipping punctuation.
    Example:
    Input:
        stanza_list = [
            [{'id': 1, 'lemma': 'I', 'text': 'I', 'upos': 'PRON'},
             {'id': [2, 3], 'lemma': None, 'text': "don't", 'upos': 'VERB'},
             {'id': 2, 'lemma': 'do', 'text': 'do', 'upos': 'AUX'},
             {'id': 3, 'lemma': "n't", 'text': "n't", 'upos': 'PART'}],
            [{'id': 1, 'lemma': 'run', 'text': 'run', 'upos': 'VERB'},
             {'id': 2, 'lemma': '.', 'text': '.', 'upos': 'PUNCT'}]
        ]
    Output (split_ngrams=True, skip_punct=True):
        ['I', 'do', "n't", 'run']
    Output (split_ngrams=False, skip_punct=False):
        ['I', "don't", 'run', '.']
    """
    lemmas = []  # List to store extracted lemmas
    ids_to_skip = []  # List to track indices of words to skip

    # Iterate over each sentence (outer lists)
    for sentence in stanza_list:
        # Iterate over each word (inner lists), represented as dictionary objects
        for word in sentence:
            # Extract the word's index
            word_index = word['id']

            # Handle n-grams (e.g., "don't" split into "do" and "n't")
            if isinstance(word_index, list):
                if not split_ngrams:
                    # If keeping joint n-grams, skip split components and store the joint word's text
                    ids_to_skip.extend(word_index)  # Mark split components for skipping
                    lemmas.append(word['text'])  # Use the full joint word text
                continue  # Move to the next word

            # Skip punctuation if skip_punct is True
            if skip_punct and word['upos'] == 'PUNCT':
                continue

            # Skip words whose indices are marked in ids_to_skip
            if word_index in ids_to_skip:
                continue

            # Add the lemma of the word to the list
            lemmas.append(word['lemma'])

    return lemmas


# TODO: find out if the lemma tokens returned form our processed data and the function above are the same
#  as in the original sample data

sandor = samplep[0]
sandor_labels = sandor["hard_labels"]
print(sandor_labels)

# TODO: the issue lies within the hard labels, they are character indices of the original text
# not word indices
# we need to convert them to word indices

# get both texts
sandor_lemmas = get_lemmas_from_stanza_list(sandor["model_output_text_processed"], skip_punct=False, split_ngrams=True)
sandor_original = sandor["model_output_text"]
print("Lemmas:", sandor_lemmas)
print("Original:", sandor_original)
print("length original:", len(sandor_original))

# now for each sequence of hard labels, print the corresponding texts of both datasets
for label_pair in sandor_labels:
    start = label_pair[0]
    end = label_pair[1]
    print(" ".join(sandor_lemmas[start:end+1]))  # we might need to remove stopwords
    print(sandor["model_output_text"][start:end+1])     # Incorrect results because it searches for character index, but do we need this?
    print("####")


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