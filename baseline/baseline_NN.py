import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocess.preprocess import timer

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def main():
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
    features = [(f"[CLS] {obj.get('query')} "
                 f"[SEP] {obj.get('lemma')} "
                 f"[SEP] UPOS: {obj.get('upos')} "
                 f"[SEP] {obj.get('xpos')}") for obj in long_data]
    labels = [obj.get('label') for obj in long_data]

    if len(features) != len(labels):
        raise Exception("The number of features and labels do not match!")

    ########################
    # MODEL TEST

    # Initialize label encoders for each feature
    features_encoder = LabelEncoder()
    # Fit the encoder
    features_encoder.fit(features)

    # Prepare the data for training
    X = [features_encoder.transform(f) for f in features]
    y = labels

    # Convert to numpy arrays for training
    X = np.array(X)
    y = np.array(y)

    print(X.shape)
    print(X)

    print(y.shape)
    print(y)

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


def test():
    pass


if __name__ == "__main__":
    main()
