import json
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.optim as optim

from preprocess.preprocess import timer
from baseline.baseline_models import HallucinationModel

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


@timer
def encode_features(features, labels):
    features_encoder = LabelEncoder()
    features_encoder.fit(features)

    X = [features_encoder.transform(f) for f in features]
    y = labels
    return np.array(X), np.array(y)


@timer
def main(input_dim: int = 3, hidden_dim1: int = 64, hidden_dim2: int = 32, output_dim: int = 2):
    features, labels = get_data_for_training()

    X, y = encode_features(features, labels)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    model = HallucinationModel(input_dim, hidden_dim1, hidden_dim2, output_dim)

    print(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    batch_size = 32

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        _, predicted = torch.max(val_outputs, 1)
        correct = (predicted == y_val).sum().item()
        accuracy = correct / y_val.size(0)
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")


def test():
    pass


if __name__ == "__main__":
    main()
