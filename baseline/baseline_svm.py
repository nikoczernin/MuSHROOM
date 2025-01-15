import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import torch

from baseline_NN_mBERT import cross_validate_model
from transformers import BertTokenizer, BertForTokenClassification
from NN_utils import Args, get_data_for_NN
from sklearn.model_selection import KFold


def train_svm(features, labels):
    """
    Trains an SVM classifier.
    - features: 2D array or matrix of input features (e.g., embeddings).
    - labels: 1D array of corresponding labels.
    """
    svm = SVC(kernel="linear", random_state=42)
    svm.fit(features, labels)
    return svm



def cross_validate_svm(features, labels, ARGS, num_folds=5):
    """
    Performs k-fold cross-validation with SVM.
    """
    labels_flat = [label[0] if label else -1 for label in labels]  # Replacing empty lists with -1
    labels_flat = np.array(labels_flat) if not isinstance(labels_flat, np.ndarray) else labels_flat

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(features)):
        print(f"Fold {fold + 1}/{num_folds}")

        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels_flat[train_idx], labels_flat[test_idx]

        svm = train_svm(X_train, y_train)

        y_pred = svm.predict(X_test)
        metrics = classification_report(y_test, y_pred, output_dict=True)
        fold_metrics.append(metrics)

    avg_metrics = {}
    for metric in fold_metrics[0].keys():
        if isinstance(fold_metrics[0][metric], dict):  # Only process class-specific metrics
            avg_metrics[metric] = np.mean([m[metric]["f1-score"] for m in fold_metrics if metric in m])

    print("Average Metrics Across Folds:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")

    return avg_metrics


def extract_embeddings(features, tokenizer, model, max_length, device):
    """
    Converts input features to BERT embeddings.
    - features: List of text inputs.
    - tokenizer: Pre-trained tokenizer for BERT.
    - model: Pre-trained BERT model.
    - max_length: Maximum sequence length.
    - device: Torch device (e.g., "cpu" or "cuda").
    """
    model.eval()
    model.to(device)
    embeddings = []

    with torch.no_grad():
        for feature in features:
            encoded = tokenizer(
                feature,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            last_hidden_state = outputs[0]

            cls_embedding = last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)

    return np.vstack(embeddings)


def training_testing_svm(ARGS=None):
    """
    Main function for training and testing with SVM.
    """
    os.chdir(os.getcwd())
    features, labels = get_data_for_NN(
        ARGS.data_path,
        max_length=ARGS.MAX_LENGTH,
        skip_overflowing_observation=ARGS.skip_overflowing_observation,
        raw_input=ARGS.raw_input,
    )
    print("Data is prepared!")

    tokenizer = BertTokenizer.from_pretrained(ARGS.TOKENIZER_MODEL_NAME)
    bert_model = BertForTokenClassification.from_pretrained(ARGS.TOKENIZER_MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)

    bert_model.resize_token_embeddings(len(tokenizer))
    device = ARGS.device
    embeddings = extract_embeddings(features, tokenizer, bert_model, ARGS.MAX_LENGTH, device)

    avg_metrics = cross_validate_svm(embeddings, labels, ARGS)
    print("SVM training and evaluation complete!")
    return avg_metrics


if __name__ == "__main__":
    args = Args()  # dont touch
    args.raw_input = True  # recommended. skips Stanza preprocessing
    # select an overflow strategy
    args.split_overflow = True
    args.truncate_overflow = False
    args.skip_overflowing_observation = False
    args.TOKENIZER_MODEL_NAME = 'bert-base-multilingual-cased'
    # Paths and names
    args.data_path = r'../data/preprocessed/val_preprocessed.json'  # input data
    args.output_path = '../data/output/val_predictions_mbert_rawinput.csv'  # location for inference output
    args.model_path = "./mbert_token_classifier_split/"  # path for saving new model or loading pretrained model
    args.DEBUG = False  # print extra information
    ##### SELECT A WORKFLOW #####
    training_testing_svm(args)

    # training_testing_cv(args)