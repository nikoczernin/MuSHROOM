from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from transformers import BertTokenizer
import numpy as np
import json

from NN_utils import NN_Args


def prepare_data(features, labels, tokenizer, max_length):
    """
    Prepare data for token-level classification with `-100` handling.
    """
    input_vectors = []
    output_labels = []
    mask = []

    for feature_seq, label_seq in zip(features, labels):
        encoded = tokenizer(
            " ".join(feature_seq),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        input_ids = encoded["input_ids"].squeeze()
        label_seq = label_seq[:max_length]

        for token_id, label in zip(input_ids, label_seq):
            input_vectors.append(token_id)
            output_labels.append(label)
            mask.append(label != -100)

    return (
        np.array(input_vectors),
        np.array(output_labels),
        np.array(mask),
    )


def train_svm_model(X_train, y_train):
    """
    Train an SVM model on the prepared input data.
    """
    print("Training SVM model...")
    svm = SVC(kernel="rbf", C=0.5, class_weight="balanced")
    svm.fit(X_train.reshape(-1, 1), y_train)
    print("SVM model training complete!")
    return svm


def evaluate_predictions(y, yhat, labels_ignore=[-100], accuracy=True, recall=True, precision=False, f1=False):
    """
    Evaluates the model's performance using precision, recall, F1-score, and accuracy.

    Inputs:
    - y: Array of true labels for response tokens.
    - yhat: Array of predicted labels for response tokens.
    - labels_ignore: List of labels to exclude during evaluation (e.g., padding token `-100`).

    Outputs:
    - metrics: Dictionary containing precision, recall, F1-score, and accuracy.
    """
    y = np.array(y)
    yhat = np.array(yhat)

    valid_indices = np.isin(y, labels_ignore, invert=True)
    y_valid = y[valid_indices]
    yhat_valid = yhat[valid_indices]

    if y_valid.size == 0:
        raise ValueError("No valid data after filtering. Ensure y contains meaningful labels.")

    # Compute evaluation metrics
    metrics = {}
    if accuracy:
        metrics["Accuracy"] = accuracy_score(y_valid, yhat_valid)
    if precision:
        metrics["Precision"] = precision_score(y_valid, yhat_valid, average="binary", zero_division=0)
    if recall:
        metrics["Recall"] = recall_score(y_valid, yhat_valid, average="binary", zero_division=0)
    if f1:
        metrics["F1"] = f1_score(y_valid, yhat_valid, average="binary", zero_division=0)

    return metrics


def evaluate_svm_model(svm, X_test, y_test, mask):
    """
    Evaluate the trained SVM model on test data.
    """
    y_pred = svm.predict(X_test.reshape(-1, 1))

    y_test_valid = y_test[mask]
    y_pred_valid = y_pred[mask]

    metrics = evaluate_predictions(
        y_test_valid,
        y_pred_valid,
        labels_ignore=[-100],
        accuracy=True,
        recall=True,
        precision=True,
        f1=True,
    )

    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


def get_data_for_svm(datapath,
                     max_length=512,
                     include_POS=False,
                     include_query=True,
                     raw_input=False):
    """
    Load and preprocess data for token-level classification.
    """
    with open(datapath) as f:
        data = json.load(f)

    features = []
    labels = []
    for sample_nr, obj in enumerate(data):
        feature_seq = []
        label_seq = []
        if include_query:
            for sentence in obj["model_input_processed"]:
                for token in sentence:
                    word_string = token.get("text") if raw_input else token.get("lemma")
                    if word_string is not None:
                        feature_seq.append(word_string)
                        label_seq.append(-100)
                        if include_POS:
                            upos = token.get("upos")
                            xpos = token.get("xpos")
                            feature_seq.extend([upos, xpos])
                            label_seq.extend([-100, -100])

        feature_seq.append("[SEP]")
        label_seq.append(-100)

        for sentence in obj["model_output_text_processed"]:
            for token in sentence:
                word_string = token.get("text", "[PAD]") if raw_input else token.get("lemma", "[PAD]")
                feature_seq.append(word_string)
                label = token.get("hallucination", 0)
                label_seq.append(int(label))
                if include_POS:
                    upos = token.get("upos", "[PAD]")
                    xpos = token.get("xpos", "[PAD]")
                    feature_seq.extend([upos, xpos])
                    label_seq.extend([-100, -100])

        if len(feature_seq) != len(label_seq):
            raise Exception(f"The number of features and labels do not match in sample {sample_nr}!")

        features.append([x if x is not None else "[PAD]" for x in feature_seq])
        labels.append([y if y is not None else -100 for y in label_seq])

    print(f"Data prepared, there are {len(features)} observations in the data.")
    return features, labels


if __name__ == "__main__":
    args = NN_Args()
    args.MAX_LENGTH = 512
    args.DATA_PATH = "../data/preprocessed/val_preprocessed.json"
    args.include_POS = True
    args.include_query = True

    features, labels = get_data_for_svm(args.DATA_PATH,
                                        max_length=args.MAX_LENGTH,
                                        include_POS=args.include_POS,
                                        include_query=args.include_query)

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    X, y, mask = prepare_data(features, labels, tokenizer, max_length=args.MAX_LENGTH)

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    mask_train, mask_test = mask[:train_size], mask[train_size:]

    X_train, y_train = X_train[mask_train], y_train[mask_train]

    svm_model = train_svm_model(X_train, y_train)

    evaluate_svm_model(svm_model, X_test, y_test, mask_test)