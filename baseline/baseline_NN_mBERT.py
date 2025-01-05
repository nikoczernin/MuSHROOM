import os
from time import time as get_time
import warnings

from transformers import AdamW
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import KFold

# import wandb

from NN_utils import *


warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

set_seed(42)


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
        if len(feature) > self.max_length:
            raise ValueError(f"Length of feature ({len(feature)}) exceeds max_length ({self.max_length})")
        if len(labels) > self.max_length:
            raise ValueError(f"Length of labels ({len(labels)}) exceeds max_length ({self.max_length})")
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
    best_total_loss = np.inf
    patience = args.patience
    stop = False
    start = get_time()
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
        if total_loss >= best_total_loss:
            if patience <= 0 and args.early_stopping:
                print(f" ====> Early stopping at epoch {epoch + 1}")
                stop = True
            else:
                patience -= 1
                print(f"Patience reduced to {patience}")
        else:
            # else reset the patience
            patience = args.patience
            # and (re)assign the previous loss to the current loss
            best_total_loss = total_loss
        if stop: break

    print(f"Training complete! Total time taken: {get_time() - start} seconds.")
    # Step 6: Save the model
    training_model.save_pretrained(args.model_path)
    args.tokenizer.save_pretrained("mbert_token_classifier")
    print("Model and tokenizer saved!")
    return epoch


def inference(inference_model, dataloader, args, flatten_output=False, binary_output=True):
    """
    Performs inference to predict token labels for the input data.
    - inference_model: Trained model for inference.
        How to use:
        for batch in trainloader:
            inference(model, batch["input_ids"])
    - dataloader: DataLoader providing batches of input data.
    - args: Arguments object specifying device for inference.
    - flatten_output: Whether to return predictions as a single concatenated array. Not recommended.

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
        # Shape: [num_obs_in_batch, size_feature_space]
        if binary_output:
            preds = torch.argmax(output.logits, dim=-1)
        else:
            preds = output.logits
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
    feature_tokens = []
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
            feature_tokens.append(tokens)
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

            if len(response_preds) != len(response_labels):
                raise Exception(f"Size of labels and predictions do not match on row {i}!")
            predicted_labels.append(response_preds)
            true_labels.append(response_labels)

    if len(true_labels) != len(predicted_labels):
        raise Exception("Size of labels and predictions do not match!")

    return feature_tokens, true_labels, predicted_labels


def cross_validate_model(features, labels, ARGS, num_folds=5):
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
        train_dataset = Subset(HallucinationDataset(features, labels, ARGS.tokenizer, ARGS.MAX_LENGTH), train_idx)
        test_dataset = Subset(HallucinationDataset(features, labels, ARGS.tokenizer, ARGS.MAX_LENGTH), test_idx)
        train_loader = DataLoader(train_dataset, batch_size=ARGS.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=ARGS.batch_size, shuffle=False)

        # Reinitialize model for each fold
        model = BertForTokenClassification.from_pretrained(ARGS.TOKENIZER_MODEL_NAME, num_labels=2)
        model.to(ARGS.device)
        # Training setup
        ARGS.optimizer = AdamW(model.parameters(), lr=ARGS.learning_rate)
        # Define optimizer and loss function
        ARGS.loss_fn = CrossEntropyLoss()
        print(f"Device: {ARGS.device}")

        # Train the model
        train_model(model, train_loader, ARGS)

        # Evaluate on the test fold
        predictions = inference(model, test_loader, ARGS)
        feature_tokens, y, yhat = get_evaluation_data(test_loader, predictions, ARGS.tokenizer)
        metrics = evaluate_predictions(y, yhat)
        fold_metrics.append(metrics)

    # Aggregate metrics across folds
    avg_metrics = {metric: np.mean([m[metric] for m in fold_metrics]) for metric in fold_metrics[0].keys()}
    print("Average Metrics Across Folds:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")

    return avg_metrics


def training_testing(ARGS=None, test=True):
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
    os.chdir(os.getcwd())
    features, labels = get_data_for_NN(ARGS.data_path, max_length=ARGS.MAX_LENGTH,
                                       split_overflow=ARGS.split_overflow,
                                       truncate_overflow=ARGS.truncate_overflow,
                                       skip_overflowing_observation=ARGS.skip_overflowing_observation,
                                       raw_input=ARGS.raw_input)
    print("Data is prepared!")
    # what is the maximum length of the features and labels?
    max_len = 0
    for i in range(len(features)):
        if len(features[i].split()) > max_len:
            max_len = len(features[i])
    for i in range(len(labels)):
        if len(labels[i]) > max_len:
            max_len = len(labels[i])
        if len(labels[i]) == 374:
            print(i, labels[i])

    print("Number of observations:", len(features), "|", len(labels))
    print("Maximum length of features or labels:", max_len)

    # Define constants
    ARGS.TOKENIZER_MODEL_NAME = "bert-base-multilingual-cased"

    print("Loading tokenizer and model...")
    ARGS.tokenizer = BertTokenizer.from_pretrained(ARGS.TOKENIZER_MODEL_NAME)
    model = BertForTokenClassification.from_pretrained(ARGS.TOKENIZER_MODEL_NAME, num_labels=2)
    print("Tokenizer and model loaded!")

    # Create datasets and dataloaders
    train = HallucinationDataset(features, labels, tokenizer=ARGS.tokenizer, max_length=ARGS.MAX_LENGTH)
    test = HallucinationDataset(features, labels, tokenizer=ARGS.tokenizer, max_length=ARGS.MAX_LENGTH)
    train_loader = DataLoader(train, batch_size=ARGS.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=ARGS.batch_size, shuffle=True)
    print("Datasets and dataloaders created!")

    # Training setup
    ARGS.optimizer = AdamW(model.parameters(), lr=ARGS.learning_rate)
    # Define optimizer and loss function
    ARGS.loss_fn = CrossEntropyLoss()
    print(f"Device: {ARGS.device}")

    if ARGS.log:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="MuSHROOM",
            # track hyperparameters and run metadata
            config={
                "dataset": ARGS.data_path,
                "architecture": "Baseline mBERT (query & response sequence classification)",
                "model": ARGS.model_path,
                "tokenizer": ARGS.TOKENIZER_MODEL_NAME,
                "device": ARGS.model_path,
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

    if ARGS.log:
        wandb.log({"num_training_epochs": num_training_epochs})
        wandb.log(metrics)
        wandb.finish()

    if not test: return

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

    feature_tokens, y, yhat = get_evaluation_data(test_loader, predictions, tokenizer=ARGS.tokenizer, DEBUG=False)

    # if required, save the prediction data to a file
    if args.output_path is not None:
        save_lists_to_delim_file(args.output_path, feature_tokens, y, yhat)

    metrics = evaluate_predictions(y, yhat)

    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


def testing(ARGS=None):
    """
    Main function to train and test the mBERT model for hallucination detection.
    - ARGS: Optional arguments object. If None, a new Args instance is created.

    Steps:
    1. Load preprocessed data
    2. Initialize tokenizer
    3. Create DataLoader for dataset.
    4. Load trained model.
    5. Perform inference on the dataset.
    6. Evaluate model performance using precision, recall, F1-score, and accuracy.
    """
    os.chdir(os.getcwd())
    features, labels = get_data_for_NN(ARGS.data_path, max_length=ARGS.MAX_LENGTH)
    print("Data is prepared!")
    print("Lengths of data:", len(features), len(labels))

    # open the json data from ARGS.data_path
    with open(ARGS.data_path, "r") as f:
        data = json.load(f)
    print("Length of JSON data:", len(data))
    # i have the suspicion that the training data has a row for each sentence in the json data
    # count the numnber of arrays in the data.model_output_text_processed
    total = 0
    for row in data:
        total += len(row["model_output_text_processed"])
    print("Total number of sentences in the JSON data:", total)

    # Define constants
    ARGS.TOKENIZER_MODEL_NAME = "bert-base-multilingual-cased"

    print("Loading tokenizer and model...")
    ARGS.tokenizer = BertTokenizer.from_pretrained(ARGS.TOKENIZER_MODEL_NAME)
    print("Tokenizer loaded!")
    # model = BertForTokenClassification.from_pretrained(ARGS.TOKENIZER_MODEL_NAME, num_labels=2)
    # TODO: Load the trained model
    print("Model loaded!")

    # Create dataset and dataloader
    test = HallucinationDataset(features, labels, tokenizer=ARGS.tokenizer, max_length=ARGS.MAX_LENGTH)
    test_loader = DataLoader(test, batch_size=8, shuffle=True)
    print("Dataset and dataloader created!")

    # Load the trained model
    model = BertForTokenClassification.from_pretrained(ARGS.model_path, num_labels=2)
    model.to(ARGS.device)

    if ARGS.log:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="MuSHROOM",
            # track hyperparameters and run metadata
            config={
                "architecture": "Baseline mBERT (query & response sequence classification)",
                "model": ARGS.model_path if ARGS.model_path is not None else ARGS.model_path,
                "tokenizer": ARGS.TOKENIZER_MODEL_NAME,
                "device": ARGS.device,
                "max_length": ARGS.MAX_LENGTH,
            }
        )

    ### TESTING
    print("Performing inference")
    predictions = inference(model, test_loader, args=ARGS)
    # extract the true labels and the predicted labels
    feature_tokens, y, yhat = get_evaluation_data(test_loader, predictions, tokenizer=ARGS.tokenizer, DEBUG=ARGS.DEBUG)

    print("size of features", len(features))
    print("size of y", len(y))

    # if required, save the prediction data to a file
    if args.output_path is not None:
        save_lists_to_delim_file(args.output_path, feature_tokens, y, yhat)

    metrics = evaluate_predictions(y, yhat)

    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    if ARGS.log:
        wandb.log(metrics)
        wandb.finish()


def training_testing_cv(ARGS=None, test=True):
    os.chdir(os.getcwd())
    features, labels = get_data_for_NN(ARGS.data_path, max_length=ARGS.MAX_LENGTH,
                                       split_overflow=ARGS.split_overflow,
                                       truncate_overflow=ARGS.truncate_overflow,
                                       skip_overflowing_observation=ARGS.skip_overflowing_observation)
    print("Data is prepared!")

    # Define constants
    ARGS.TOKENIZER_MODEL_NAME = "bert-base-multilingual-cased"
    print("Loading tokenizer and model...")
    ARGS.tokenizer = BertTokenizer.from_pretrained(ARGS.TOKENIZER_MODEL_NAME)
    print("Tokenizer and model loaded!")
    print("Starting cross-validation...")
    avg_metrics = cross_validate_model(
        features=features,
        labels=labels,
        ARGS=ARGS,
        num_folds=5
    )

    if ARGS.log:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="MuSHROOM",
            # track hyperparameters and run metadata
            config={
                "dataset": ARGS.data_path,
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

    # Print average metrics across all folds
    print("Cross-validation completed. Average Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    args = Args()
    args.raw_input = True
    args.split_overflow = True
    args.truncate_overflow = False
    args.skip_overflowing_observation = False
    args.data_path = '../data/preprocessed/val_preprocessed.json'
    args.output_path = '../data/output/val_predictions_mbert_rawinput.csv'
    args.model_path = "./mbert_token_classifier_split/"
    args.DEBUG = False
    # training_testing(args)
    testing(args)
    # training_testing_cv(args)
