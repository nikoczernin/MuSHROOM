import os

import numpy as np

from transformers import AdamW
from transformers import BertTokenizer, BertForTokenClassification
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold

from baseline_utils import get_data_for_training

from time import time as get_time
# import wandb

import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


# Helper class for all hyperparameters
class Args:
    def __init__(self):
        self.MAX_LENGTH = 128 * 2  # Max token length for mBERT
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epochs = 100
        self.patience = 3
        self.early_stopping = True
        self.TOKENIZER_MODEL_NAME = None
        self.tokenizer = None
        self.optimizer = None
        self.loss_fn = None
        self.model_name = None
        self.data_path = None
        self.learning_rate = 2e-5
        self.log = False


# Dataset class to handle features and labels for token classification
class HallucinationDataset(Dataset):
    def __init__(self, features, labels, tokenizer, max_length):
        """
        Initializes the dataset.
        - features: List of input texts (queries and responses).
        - labels: List of binary labels aligned with tokens in the responses.
        - tokenizer: Pre-trained tokenizer to tokenize the input text.
        - max_length: Maximum token length for padding/truncation.
        """
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns a single data point (tokenized text and aligned labels).
        """
        feature = self.features[idx]
        labels = self.labels[idx]

        # Tokenize input text
        encoded = self.tokenizer(
            feature,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # label_ids starts as a list of -100 values, matching the length of the tokenized input sequence.
        label_ids = [-100] * len(encoded["input_ids"][0])  # Ignore non-response tokens
        # the self.tokenizer.sep_token_id identifies the [SEP] token, which separates the query from the response
        # response_start points to the position in the tokenized sequence immediately after the first [SEP],
        # marking the start of the response
        response_start = encoded["input_ids"][0].tolist().index(self.tokenizer.sep_token_id) + 1
        # labels is the binary label list for the response tokens (e.g., [0, 0, 1, 0, 1]).
        # This line replaces the corresponding portion of label_ids (starting from response_start) with
        # the actual labels for the response tokens.
        label_ids[response_start:response_start + len(labels)] = labels

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label_ids, dtype=torch.long),
        }


def train_model(training_model, dataloader, args):
    """
    Trains the mBERT model for token classification.
    - training_model: Pre-trained mBERT model for fine-tuning.
    - dataloader: DataLoader providing batches of training data.
    - args: Arguments object with training configurations (device, optimizer, etc.).
    """
    training_model.to(args.device)
    print("Model moved to device!")
    # Step 5: Training loop
    training_model.train()
    print("Training started!")
    previous_loss = np.inf
    patience = args.patience
    stop = False
    for epoch in range(args.max_epochs):  # Maximum number of epochs
        t = get_time()
        total_loss = 0
        for batch in dataloader:
            # Move data to device
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["label"].to(args.device)
            # Forward pass
            outputs = training_model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     labels=labels)
            loss = outputs.loss
            # Backward pass
            args.optimizer.zero_grad()
            loss.backward()
            args.optimizer.step()
            total_loss += loss.item()

        cur_time = get_time()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}, Time: {cur_time - t}")
        if args.log:
            wandb.log({"epoch": epoch, "loss": total_loss / len(dataloader), "time": cur_time - t})

        # if the loss did not improve, test the patience
        if total_loss >= previous_loss:
            if patience <= 0 and args.early_stopping:
                print(f" ====> Early stopping at epoch {epoch + 1}")
                stop = True
            else:
                patience -= 1
                print(f"Patience reduced to {patience}")
        else:
            # else reset the patience
            patience = args.patience
        # (re)assign the previous loss to the current loss
        previous_loss = total_loss
        if stop: break

    print("Training complete!")
    # Step 6: Save the model
    training_model.save_pretrained(args.model_name)
    args.tokenizer.save_pretrained("mbert_token_classifier")
    print("Model and tokenizer saved!")
    return epoch


def inference(inference_model, dataloader, args, flatten_output=False):
    """
    Performs inference to predict token labels for the input data.
    - inference_model: Trained model for inference.
        How to use:
        for batch in trainloader:
            inference(model, batch["input_ids"])
    - dataloader: DataLoader providing batches of input data.
    - args: Arguments object specifying device for inference.
    - flatten_output: Whether to return predictions as a single concatenated array.

    Returns:
    - all_predictions: Predicted labels for each token in the input.
    """
    inference_model.eval()  # is this necessary?
    all_predictions = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        # Get logits from the model
        output = inference_model(input_ids=input_ids, attention_mask=attention_mask)
        # TokenClassifierOutput with attrs: loss, logits, grad_fn, hidden_states, attentions
        # the logits should be the probabilities of all classes for each observation
        # and therefore have shape (num_obs_in_batch, size_feature_space, num_class_labels)
        # Convert logits to predicted labels
        preds = torch.argmax(output.logits, dim=-1)  # Shape: [num_obs_in_batch, size_feature_space]
        all_predictions.append(preds.cpu())
    if flatten_output:
        all_predictions = torch.cat(all_predictions, dim=0)  # Shape: (total_num_samples, seq_len)
    return all_predictions


def get_evaluation_data(dataloader, predictions, tokenizer, DEBUG=False, include_padding=False):
    """
        Processes predictions and true labels from a dataloader to extract token-level evaluation data.

        Inputs:
        - dataloader: PyTorch DataLoader providing batches of input data, including:
            * 'input_ids': Tokenized input sequences (query + response).
            * 'label': Ground truth labels for each token, aligned with the tokenized sequence.
        - predictions: 2D array (or tensor) of model predictions, where each row corresponds to a sequence
                       and each column is a predicted label for a token.
        - tokenizer: Tokenizer used for decoding tokens.
        - DEBUG: (bool) Optional flag to enable detailed debugging output for inspection.
        - include_padding: Whether to include padding tokens in evaluation.

        Outputs:
        - true_labels: A list of arrays, where each array contains the ground truth labels for the response tokens.
                       Excludes padding and query tokens.
        - predicted_labels: A list of arrays, where each array contains the predicted labels for the response tokens.
                            Excludes padding and query tokens.

        Purpose:
        - Aligns predictions with their respective tokens and labels in the response part of the sequence.
        - Filters out padding (`-100` labels) and query tokens, focusing only on response tokens.
        - Returns data ready for token-level evaluation metrics (e.g., precision, recall, F1).
    """
    true_labels = []  # Store true labels for evaluation
    predicted_labels = []  # Store predicted labels for evaluation
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        labels = batch["label"]
        # Move labels and predictions to CPU for processing
        batch_labels = labels.numpy()  # Shape (batch size, size feature space)
        batch_preds = predictions[batch_idx].numpy()  # Shape (batch size, size feature space)

        # Iterate over each sample in the batch
        for i in range(input_ids.shape[0]):
            # Get the tokenized text
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
            if DEBUG: print(f"Current row: {i + (batch_idx + 1) * input_ids.shape[0]}")
            if DEBUG: print(f"\tTokens in row:", tokens)
            # Locate response tokens (after first [SEP])
            sep_index = input_ids[i].tolist().index(tokenizer.sep_token_id) + 1
            response_tokens = tokens[sep_index:]
            if DEBUG: print("\tTokens in response:", response_tokens)
            response_preds = batch_preds[i][sep_index:]
            if DEBUG: print("\tPredicted labels for response:", response_preds)
            response_labels = batch_labels[i][sep_index:]
            if DEBUG: print("\tActual labels for response:", response_labels)
            if not include_padding:
                # now filter out the labels with the -100 labels, those are just padding
                padding_index = response_labels.tolist().index(-100)
                if DEBUG: print("\tPadding starts at index", padding_index)
                response_preds = response_preds[:padding_index]
                response_labels = response_labels[:padding_index]
                if DEBUG: print("\tPredicted labels truncated for response:", response_preds)
                if DEBUG: print("\tActual labels truncated for response:", response_labels)
            predicted_labels.append(response_preds)
            true_labels.append(response_labels)

    if len(true_labels) != len(predicted_labels):
        raise Exception("Size of labels and predictions do not match!")
    return true_labels, predicted_labels


def evaluate_model(y, yhat, labels_ignore=[-100]):
    """
    Evaluates the model's performance using precision, recall, F1-score, and accuracy.

    Inputs:
    - y: List of arrays with true labels for response tokens.
    - yhat: List of arrays with predicted labels for response tokens.
    - labels_ignore: List of labels to exclude during evaluation (e.g., padding token `-100`).

    Outputs:
    - metrics: Dictionary containing precision, recall, F1-score, and accuracy.
    """
    # Flatten lists of arrays
    y_flat = np.concatenate(y).flatten()
    yhat_flat = np.concatenate(yhat).flatten()

    # Filter out ignored labels (like padding)
    valid_indices = np.isin(y_flat, labels_ignore, invert=True)
    y_filtered = y_flat[valid_indices]
    yhat_filtered = yhat_flat[valid_indices]

    # Compute metrics
    precision = precision_score(y_filtered, yhat_filtered, average="binary")
    recall = recall_score(y_filtered, yhat_filtered, average="binary")
    f1 = f1_score(y_filtered, yhat_filtered, average="binary")
    accuracy = accuracy_score(y_filtered, yhat_filtered)

    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Accuracy": accuracy,
    }
    return metrics


def cross_validate_model(features, labels, tokenizer, model, args, num_folds=5):
    """
    Performs k-fold cross-validation on the dataset.

    Inputs:
    - features: List of text inputs.
    - labels: List of corresponding token labels.
    - tokenizer: Pre-trained tokenizer for text processing.
    - model: Pre-trained token classification model.
    - args: Arguments object with training configuration.
    - num_folds: Number of folds for cross-validation.

    Outputs:
    - avg_metrics: Dictionary with average metrics across folds.
    """
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(features)):
        print(f"Fold {fold + 1}/{num_folds}")

        # Create train/test datasets and dataloaders
        train_dataset = Subset(HallucinationDataset(features, labels, tokenizer, args.max_length), train_idx)
        test_dataset = Subset(HallucinationDataset(features, labels, tokenizer, args.max_length), test_idx)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Reinitialize model for each fold
        model_copy = BertForTokenClassification.from_pretrained(args.model_name, num_labels=2)
        model_copy.to(args.device)
        optimizer = AdamW(model_copy.parameters(), lr=args.learning_rate)

        # Train the model
        train_model(model_copy, train_loader, args)

        # Evaluate on the test fold
        predictions = inference(model_copy, test_loader, args, flatten_output=True)
        y, yhat = get_evaluation_data(test_loader, predictions, tokenizer)
        metrics = evaluate_model(y, yhat)
        fold_metrics.append(metrics)

    # Aggregate metrics across folds
    avg_metrics = {metric: np.mean([m[metric] for m in fold_metrics]) for metric in fold_metrics[0].keys()}
    print("Average Metrics Across Folds:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")

    return avg_metrics


def conduct_test(ARGS=None):
    """
    Main function to train and test the mBERT model for hallucination detection.
    - ARGS: Optional arguments object. If None, a new Args instance is created.

    Steps:
    1. Load preprocessed data.
    2. Initialize tokenizer and model.
    3. Create DataLoader for training and testing datasets.
    4. Train the model.
    5. Perform inference on the test dataset.
    6. Evaluate model performance using precision, recall, F1-score, and accuracy.
    """
    # TODO: set the cwd to this file
    os.chdir(os.getcwd())
    features, labels = get_data_for_training(ARGS.data_path)
    print("Data is prepared!")

    # Define constants
    ARGS.TOKENIZER_MODEL_NAME = "bert-base-multilingual-cased"


    print("Loading tokenizer and model...")
    ARGS.tokenizer = BertTokenizer.from_pretrained(ARGS.TOKENIZER_MODEL_NAME)
    model = BertForTokenClassification.from_pretrained(ARGS.TOKENIZER_MODEL_NAME, num_labels=2)
    print("Tokenizer and model loaded!")

    # Create datasets and dataloaders
    train = HallucinationDataset(features, labels, tokenizer=ARGS.tokenizer, max_length=ARGS.MAX_LENGTH)
    test = HallucinationDataset(features, labels, tokenizer=ARGS.tokenizer, max_length=ARGS.MAX_LENGTH)
    train_loader = DataLoader(train, batch_size=8, shuffle=True)
    test_loader = DataLoader(test, batch_size=8, shuffle=True)
    print("Datasets and dataloaders created!")

    # Training setup
    ARGS.optimizer = AdamW(model.parameters(), lr=ARGS.learning_rate)
    # Define optimizer and loss function
    ARGS.loss_fn = CrossEntropyLoss()
    print(f"Device: {ARGS.device}")
    ARGS.model_name = "mbert_token_classifier"

    if ARGS.log:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="MuSHROOM",
            # track hyperparameters and run metadata
            config={
                "dataset": "sample",
                "architecture": "Baseline mBERT (query & response sequence classification)",
                "model": ARGS.model_name,
                "tokenizer": ARGS.TOKENIZER_MODEL_NAME,
                "device": ARGS.model_name,
                "loss_function": ARGS.loss_fn,
                "optimizer": ARGS.optimizer,
                "max_length": ARGS.MAX_LENGTH,
                "max_epochs": ARGS.max_epochs,
                "patience": ARGS.patience,
                "learning_rate": ARGS.learning_rate,
            }
        )

    ### TRAINING
    num_training_epochs = train_model(model, train_loader, args=ARGS)

    ### TESTING
    print("Performing inference")
    predictions = inference(model, test_loader, args=ARGS)
    # The shape (total_samples, seq_len) arises because each input to
    # the model is tokenized and padded/truncated to a fixed length
    # (max_length, typically 128 or another specified value)
    # The model predicts a label for every token in the sequence, including:
    #  - Query tokens
    #  - Response tokens
    #  - Padding tokens
    # How do we find the response tokens though?
    # locate the [SEP] token, which separates the query from the response
    # Tokens after the first [SEP] up to the end of the response are the ones you care about
    # Exclude. Padding tokens & Special tokens like [CLS] and [SEP]

    y, yhat = get_evaluation_data(test_loader, predictions, tokenizer=ARGS.tokenizer, DEBUG=False)
    print(y[0])
    print(yhat[0])

    # if required, save the prediction data to a file
    if args.output_path is not None:
        with open(args.output_path, "w") as f:
            f.write("True;Predicted\n")
            for i in range(len(y)):
                f.write(f"{y[i]};{yhat[i]}\n")

    metrics = evaluate_model(y, yhat)

    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    if ARGS.log:
        wandb.log({"num_training_epochs": num_training_epochs})
        wandb.log(metrics)
        wandb.finish()

    '''
    ARGS.batch_size = 8
    ARGS.learning_rate = 2e-5
    ARGS.max_length = 128
    ARGS.model_name = TOKENIZER_MODEL_NAME
    
    print("Starting cross-validation...")
    avg_metrics = cross_validate_model(
        features=features,
        labels=labels,
        tokenizer=tokenizer,
        model=model,
        args=ARGS,
        num_folds=5
    )
    
    # Print average metrics across all folds
    print("Cross-validation completed. Average Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    '''


if __name__ == "__main__":
    args = Args()
    args.data_path = '../data/preprocessed/val_preprocessed.json'
    args.data_path = '../data/output/val_predictions_mbert1.csv'
    conduct_test(args)
