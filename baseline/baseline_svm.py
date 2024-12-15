import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from baseline.baseline_utils import timer, get_data_for_NN

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
def train_svm(X_train, y_train, kernel='linear', C=1.0):
    """
    Trains an SVM classifier.

    Parameters:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        kernel (str): Kernel type for SVM.
        C (float): Regularization parameter.

    Returns:
        model: Trained SVM model.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

@timer
def evaluate_svm(model, scaler, X_test, y_test):
    """
    Evaluate the trained SVM model on a test dataset.

    Parameters:
        model: Trained SVM model.
        scaler: Scaler used for feature normalization.
        X_test (np.ndarray): Test set features.
        y_test (np.ndarray): Test set labels.

    Returns:
        None
    """
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

@timer
def main(kernel='linear', C=1.0, test_size=0.2):
    """
    Main function to train and evaluate the SVM-based model.

    Parameters:
        kernel (str): Kernel type for SVM.
        C (float): Regularization parameter.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        None
    """
    # Load and preprocess data
    features, labels = get_data_for_NN(r"data\preprocessed\sample_preprocessed.json")
    X = encode_features(features)
    y = np.array(labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train the SVM
    model, scaler = train_svm(X_train, y_train, kernel=kernel, C=C)

    # Evaluate the SVM
    evaluate_svm(model, scaler, X_test, y_test)

if __name__ == "__main__":
    main(kernel='rbf', C=1.0, test_size=0.2)