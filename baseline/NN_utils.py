import json
from functools import wraps
import time
import torch
import numpy as np
import random

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



# Helper class for all hyperparameters
class Args:
    def __init__(self):
        self.MAX_LENGTH = 128 * 2  # Max token length for mBERT
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Training time handling
        self.max_epochs = 100
        self.patience = 3
        self.early_stopping = True
        self.learning_rate = 2e-5
        self.batch_size = 8

        # if this is true, the model will be trained and tested on the unprocessed input data
        self.raw_input = False

        # if true, POS tokens will be used in input data, they iwll be ignored by optimizer though
        self.include_POS = False

        self.TOKENIZER_MODEL_NAME = None
        self.tokenizer = None
        self.optimizer = None
        self.loss_fn = None
        self.model_name = None
        # input data path
        self.data_path = None
        # Wandb logging
        self.log = False
        self.DEBUG = False
        # Handling of data size exceeding the maximum length
        self.split_overflow = True
        self.truncate_overflow = False
        self.skip_overflowing_observation = False


def timer(func):
    """
    A decorator to measure and print the execution time of a function.

    Args:
    - func (function): The function to be wrapped by the timer decorator.

    Returns:
    - wrapper (function): A wrapped function that calculates and prints the time
                           taken to execute the original function.

    This decorator can be used to wrap functions and output their execution time
    in seconds.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"{func.__name__} executed in {duration:.4f} seconds")
        return result

    return wrapper


@timer
def get_data_for_NN(datapath,
                    max_length=512,
                    include_POS=False,
                    truncate_overflow=True,
                    skip_overflowing_query=False,
                    skip_overflowing_observation=False,
                    split_overflow=False,
                    raw_input=False
                    ):
    # for each observation in the data, we want to create two training objects, saved in separate lists
    # 1: the features
    # 2: the labels
    # the features are the query and the response, prepended by a [CLS] token and separated by a [SEP] token
    # the labels are a list of binary labels (1 for hallucatinations)

    # read the json of the preprocessed data
    with open(datapath) as f:
        data = json.load(f)

    features = []
    labels = []
    for sample_nr, obj in enumerate(data):
        # now create a new object for each preprocessing mode output token and append it to the long_data list
        # we iterate over the processed token objects, with the iterating number being i
        # the ith token should correspond with the ith label
        # the label of this response is a list of binary labels (1 for hallucatinations)

        feature_seq = []
        label_seq = []
        # add the query tokens and labels
        for sentence in obj["model_input_processed"]:
            for token in sentence:
                # add every lemma or original text as a feature
                if raw_input: word_string = token.get("text")
                else: word_string = token.get("lemma")
                if word_string is not None:
                    feature_seq.append(word_string)
                    # for each word add also the label to the labels sequence
                    label = -100
                    label_seq.append(int(label) if label is not None else 0)
                    if include_POS:
                        # raise Exception("POS-tags are not yet implemented")
                        # these 4 rows are optional: include POS-tags in the features
                        # also add a "ignore" label to the label_sequence
                        upos = token.get("upos")
                        xpos = token.get("xpos")
                        feature_seq.append(upos)
                        feature_seq.append(xpos)
                        # add "ignore" tokens for the labels here
                        label_seq.append(-100)
                        label_seq.append(-100)
        # add a separator token to the sequences
        feature_seq.append("[SEP]")
        label_seq.append(-100)
        # add the response tokens and labels
        for sentence in obj["model_output_text_processed"]:
            for token in sentence:
                # add every lemma or original text as a feature
                if raw_input: word_string = token.get("text", "[PAD]")
                else: word_string = token.get("lemma", "[PAD]")
                feature_seq.append(word_string)
                # for each word add also the label to the labels sequence
                label = token.get("hallucination", 0) # label: default 0, if not hallucination
                label_seq.append(int(label))
                if include_POS:
                    # raise Exception("POS-tags are not yet implemented")
                    # these 4 rows are optional: include POS-tags in the features
                    # also add a "ignore" label to the label_sequence
                    upos = token.get("upos", "[PAD]")
                    xpos = token.get("xpos", "[PAD]")
                    feature_seq.append(upos)
                    feature_seq.append(xpos)
                    # add "ignore" tokens for the labels here
                    label_seq.append(-100)
                    label_seq.append(-100)

        # add a separator token to the sequences
        # save the data in the lists (Nones are not allowed here)
        # print(feature_seq)
        # print(label_seq)
        features.append([x if x is not None else "[PAD]" for x in feature_seq])
        labels.append([y if y is not None else -100 for y in label_seq])

    if len(features) != len(labels):
        raise Exception("The number of features and labels do not match!")

    # join the features into a single string
    features = [" ".join(feature_seq) for feature_seq in features]
    return features, labels


def set_seed(seed: int):
    """
    Sets the seed for reproducibility across multiple libraries.
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)  # Set seed for Python's built-in random module
    np.random.seed(seed)  # Set seed for NumPy
    torch.manual_seed(seed)  # Set seed for PyTorch on CPU
    torch.cuda.manual_seed(seed)  # Set seed for PyTorch on the current GPU
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs (if using multiple GPUs)

    # Ensure deterministic behavior for reproducible results
    torch.backends.cudnn.deterministic = True  # Forces CUDA to use deterministic algorithms
    torch.backends.cudnn.benchmark = False  # Disables optimization that can introduce randomness


def evaluate_predictions(y, yhat, labels_ignore=[-100]):
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
    return all_predictions



def save_lists_to_delim_file(output_path, *args, delimiter="[DELIM]"):
    # *args are an unknown number of lists with the same length
    # write a csv file where the ith row is the ith element of each list
    with open(output_path, "w") as f:
        for i in range(len(args[0])):
            row = [str(arg[i]).replace("\n", "") for arg in args]
            f.write(f"{delimiter}".join(row) + "\n")

