import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn

from baseline.baseline_utils import timer, get_data_for_training
from baseline.baseline_models import HallucinationBaselineNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


@timer
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
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    model = HallucinationBaselineNN(input_dim, hidden_dim1, hidden_dim2, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on a test dataset.

    Parameters:
        model (nn.Module): Trained PyTorch model.
        X_test (torch.Tensor): Test set features.
        y_test (torch.Tensor): Test set labels.

    Returns:
        None
    """
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        outputs = model(X_test)  # Forward pass
        _, predictions = torch.max(outputs, 1)  # Get class predictions

    # Convert predictions and labels to NumPy arrays
    y_pred = predictions.numpy()
    y_true = y_test.numpy()

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Print metrics
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)


@timer
def main(hidden_dim1: int = 64,
         hidden_dim2: int = 32,
         output_dim: int = 2,
         epochs: int = 20,
         batch_size: int = 32,
         learning_rate: float = 0.001):
    features, labels = get_data_for_training()
    X = encode_features(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    model = train_model(
        X_train_tensor, y_train_tensor,
        input_dim=X_train.shape[1],
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        output_dim=output_dim,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    evaluate_model(model, X_test_tensor, y_test_tensor)


def test():
    pass


if __name__ == "__main__":
    main(epochs=20,
         batch_size=32,
         learning_rate=0.005
         )
