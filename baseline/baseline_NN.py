import json
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn

from preprocess.preprocess import timer
from baseline.baseline_models import HallucinationBaselineNN

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


@timer
def get_data_for_training():
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

    return features, labels


def parse_feature(feature):
    """
    Parses a feature string into its components: query, word, UPOS, and XPOS.

    Parameters:
        feature (str): Input string in the structured format.

    Returns:
        tuple: Parsed components (query, word, UPOS, XPOS).
    """
    parts = feature.split(" [SEP] ")
    query = parts[0].replace("[CLS] ", "").strip()
    word = parts[1].strip()
    upos = parts[2].replace("UPOS: ", "").strip()
    xpos = parts[3].strip()
    return query, word, upos, xpos


@timer
def encode_features(features):
    """
    Encodes features into numeric values by processing their components.

    Parameters:
        features (list of str): List of features in structured format.

    Returns:
        np.ndarray: Encoded features as numeric values.
    """
    queries, words, upos_list, xpos_list = [], [], [], []

    for feature in features:
        query, word, upos, xpos = parse_feature(feature)
        queries.append(query)
        words.append(word)
        upos_list.append(upos)
        xpos_list.append(xpos)

    query_encoder = LabelEncoder()
    word_encoder = LabelEncoder()
    upos_encoder = LabelEncoder()
    xpos_encoder = LabelEncoder()

    query_encoded = query_encoder.fit_transform(queries)
    word_encoded = word_encoder.fit_transform(words)
    upos_encoded = upos_encoder.fit_transform(upos_list)
    xpos_encoded = xpos_encoder.fit_transform(xpos_list)

    encoded_features = np.stack((query_encoded, word_encoded, upos_encoded, xpos_encoded), axis=1)
    return encoded_features


@timer
def train_model(X_tensor, y_tensor, input_dim, hidden_dim1, hidden_dim2, output_dim, num_epochs=10, batch_size=32, learning_rate=0.001):
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Model, Loss, Optimizer
    model = HallucinationBaselineNN(input_dim, hidden_dim1, hidden_dim2, output_dim)
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        val_accuracy = correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model


@timer
def main(input_dim: int = 3, hidden_dim1: int = 64, hidden_dim2: int = 32, output_dim: int = 2):
    features, labels = get_data_for_training()

    X = encode_features(features)
    y = np.array(labels)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = train_model(
        X_tensor, y_tensor,
        input_dim=X.shape[1],
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        output_dim=output_dim,
        num_epochs=20,
        batch_size=32,
        learning_rate=0.001
    )


def test():
    pass


if __name__ == "__main__":
    main()
