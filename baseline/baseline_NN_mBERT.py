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
        - features: List of input text lists (including tokens from queries and responses).
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
        original_feature = self.features[idx]
        original_labels = self.labels[idx]
        # Problem: this here tokenizer sometimes splits words that out preprocessing did in fact not split
        # Consequence: if there are suddenly more tokens than before, the labels will not align anymore
        # Solution: iterate over every ith token in n tokens in the preprocessed data and tokenize it manually
        # then every ith token will be a list of m tokens
        # save each of them into tokenized_tokens (OPTIONAL)
        # then duplicate the ith label m times
        tokenized_tokens = []
        # the aligned labels need to start with -100 because the first token is always [CLS]
        aligned_labels = [-100]
        for i, original_token in enumerate(original_feature):
            for tokenized_token in self.tokenizer.tokenize(original_token):
                # tokenized_tokens.append(tokenized_token)
                aligned_labels.append(original_labels[i])

        # Pad or truncate both sequences to max_length
        # tokenized_tokens = tokenized_tokens[:self.max_length] + ["[PAD]"] * (self.max_length - len(tokenized_tokens))
        aligned_labels = aligned_labels[:self.max_length] + [-100] * (self.max_length - len(aligned_labels))

        # # Tokenize input text
        encoded = self.tokenizer(
            " ".join(original_feature),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(aligned_labels, dtype=torch.long),
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
            # tokens sometimes get split up by the tokenizer
            # for example, "Fußball" becomes ["Fu", "##ß", "##ball"]
            # join them back together
            if DEBUG: print(f"Current row: {i + (batch_idx + 1) * input_ids.shape[0]}")
            if DEBUG: print(f"\tTokens in row ({len(tokens)}):", tokens)
            if DEBUG: print(f"\tPredicted labels in row ({len(batch_preds[i])}):", batch_preds[i])
            if DEBUG: print(f"\tActual labels in row ({len(batch_labels[i])}):", batch_labels[i])

            # Locate response tokens (after first [SEP])
            sep_indices = [i for i, token in enumerate(tokens) if token == "[SEP]"]
            if DEBUG: print("\tIndices of [SEP] tokens:", sep_indices)
            # there should be 2 sep indices
            if len(sep_indices) != 2:
                if DEBUG:
                    raise Exception(f"Expected 2 [SEP] tokens, found {len(sep_indices)}!")
                else:
                    continue
            response_tokens = tokens[sep_indices[0] + 1: sep_indices[1]]
            response_preds = batch_preds[i][sep_indices[0] + 1: sep_indices[1]]
            response_labels = batch_labels[i][sep_indices[0] + 1: sep_indices[1]]
            if DEBUG: print(f"\tTokens in response ({len(response_tokens)}):", response_tokens)
            if DEBUG: print(f"\tPredicted labels for response ({len(response_preds)}):", response_preds)
            if DEBUG: print(f"\tActual labels for response ({len(response_labels)}):", response_labels)
            if DEBUG: print("\n------------------------------------\n")

            if len(response_preds) != len(response_labels):
                if DEBUG:
                    raise Exception(f"Size of labels and predictions do not match on row {i}!")
                else:
                    continue
            feature_tokens.append(tokens)
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
    features, labels = get_data_for_NN(ARGS.data_path,
                                       max_length=ARGS.MAX_LENGTH,
                                       include_POS=ARGS.include_POS,
                                       include_query=ARGS.include_query,
                                       skip_overflowing_observation=ARGS.skip_overflowing_observation,
                                       raw_input=ARGS.raw_input)
    print("Data is prepared!")
    # what is the maximum length of the features and labels?
    max_len = 0
    for i in range(len(features)):
        if len(features[i]) > max_len:
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

    ### TRAINING
    num_training_epochs = train_model(model, train_loader, args=ARGS)

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
    features, labels = get_data_for_NN(ARGS.data_path,
                                       max_length=ARGS.MAX_LENGTH,
                                       include_query=ARGS.include_query,
                                       include_POS=ARGS.include_POS
                                       )
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

    # Create dataset and dataloader
    test = HallucinationDataset(features, labels, tokenizer=ARGS.tokenizer, max_length=ARGS.MAX_LENGTH)
    test_loader = DataLoader(test, batch_size=8, shuffle=True)
    print("Dataset and dataloader created!")

    # Load the trained model
    model = BertForTokenClassification.from_pretrained(ARGS.model_path, num_labels=2)
    model.to(ARGS.device)
    print("Model loaded!")

    ### TESTING
    print("Performing inference")
    predictions = inference(model, test_loader, args=ARGS)
    # extract the true labels and the predicted labels
    feature_tokens, y, yhat = get_evaluation_data(test_loader, predictions, tokenizer=ARGS.tokenizer, DEBUG=ARGS.DEBUG)

    # if required, save the prediction data to a file
    if args.output_path is not None:
        save_lists_to_delim_file(args.output_path, feature_tokens, y, yhat)

    metrics = evaluate_predictions(y, yhat)

    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


def training_testing_cv(ARGS=None, test=True):
    os.chdir(os.getcwd())
    features, labels = get_data_for_NN(ARGS.data_path,
                                       max_length=ARGS.MAX_LENGTH,
                                       include_POS=ARGS.include_POS,
                                       include_query=ARGS.include_query,
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

    # Print average metrics across all folds
    print("Cross-validation completed. Average Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    args = Args()  # dont touch
    args.MAX_LENGTH = 128
    args.raw_input = True  # recommended. skips Stanza preprocessing
    args.include_POS = False  # include POS tags in the input data for more context
    args.include_query = True  # include the query in the input
    args.skip_overflowing_observation = True  # skip observations that exceed the max length instead of truncating them
    # Paths and names
    args.data_path = '../data/preprocessed/val_preprocessed.json'  # input data
    args.output_path = '../data/output/val_predictions_mbert_skip.csv'  # location for inference output
    args.model_path = "./mbert_token_classifier_skip/"  # path for saving new model or loading pretrained model
    args.DEBUG = True  # print extra information
    ##### SELECT A WORKFLOW #####
    # training_testing(args) # train a new model and test it once
    testing(args)  # load a pretrained model and test it once
    # training_testing_cv(args) # train k-fold cross validation and test once
