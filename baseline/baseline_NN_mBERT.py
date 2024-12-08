import json
import torch
from transformers import AdamW
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification
from torch.utils.data import DataLoader, Dataset
from baseline_utils import get_data_for_training

import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


# create a class for the dataset
class HallucinationDataset(Dataset):
    def __init__(self, features, labels, tokenizer, max_length):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
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


def train_model(training_model, dataloader, args, tokenizer_name="mbert_token_classifier"):
    training_model.to(args.device)
    print("Model moved to device!")
    # Step 5: Training loop
    training_model.train()
    print("Training started!")
    for epoch in range(args.num_epochs):  # Number of epochs
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
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
    print("Training complete!")
    # Step 6: Save the model
    training_model.save_pretrained(args.model_name)
    tokenizer.save_pretrained("mbert_token_classifier")
    print("Model and tokenizer saved!")


def inference(inference_model, dataloader, flatten_output=False):
    # inputs: input_ids of a batch, which is the yield of iterating the trainloader
    # for batch in trainloader:
    #     inference(model, batch["input_ids"])
    # returns:
    inference_model.eval()  # is this necessary?
    all_predictions = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(ARGS.device)
        attention_mask = batch["attention_mask"].to(ARGS.device)
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


def get_evaluation_data(dataloader, predictions, DEBUG=False, include_padding=False):
    """
        Processes predictions and true labels from a dataloader to extract token-level evaluation data.

        Inputs:
        - dataloader: PyTorch DataLoader providing batches of input data, including:
            * 'input_ids': Tokenized input sequences (query + response).
            * 'label': Ground truth labels for each token, aligned with the tokenized sequence.
        - predictions: 2D array (or tensor) of model predictions, where each row corresponds to a sequence
                       and each column is a predicted label for a token.
        - DEBUG: (bool) Optional flag to enable detailed debugging output for inspection.

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


features, labels = get_data_for_training('../data/preprocessed/sample_preprocessed.json')
print("Data is prepared!")

# Define constants
TOKENIZER_MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 128  # Max token length for mBERT

print("Loading tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)
# model = BertForSequenceClassification.from_pretrained(TOKENIZER_MODEL_NAME, num_labels=2)
model = BertForTokenClassification.from_pretrained(TOKENIZER_MODEL_NAME, num_labels=2)
print("Tokenizer and model loaded!")

# Create datasets and dataloaders
train = HallucinationDataset(features, labels, tokenizer, max_length=MAX_LENGTH)
test = HallucinationDataset(features, labels, tokenizer, max_length=MAX_LENGTH)
train_loader = DataLoader(train, batch_size=8, shuffle=True)
test_loader = DataLoader(test, batch_size=8, shuffle=True)
print("Datasets and dataloaders created!")


# Training setup
# Helper class for neural network hyper-parameters
class Args:
    pass


ARGS = Args()
ARGS.optimizer = AdamW(model.parameters(), lr=2e-5)
# Define optimizer and loss function
ARGS.loss_fn = CrossEntropyLoss()
# Set device
ARGS.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {ARGS.device}")
ARGS.model_name = "mbert_token_classifier"
ARGS.num_epochs = 3
### TRAINING
train_model(model, train_loader, args=ARGS)

### TESTING
print("we testing now")

predictions = inference(model, test_loader)
# print(predictions)
# print(predictions.shape)
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

y, yhat = get_evaluation_data(test_loader, predictions, DEBUG=False)
print(y[0])
print(yhat[0])
# TODO: apply evaluation metrics here!
