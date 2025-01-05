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
def get_data_for_NN(datapath, include_POS=False,
                    max_length=512,
                    ignore_label=-100,
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

    # TODO: often the data is too long to make for a reasonable input to the model
    # we can truncate the data to a maximum length, but this will result in a loss of information
    # we can also split the data in multiple observations, but this will result in a loss of context
    # if the query is too long, we can ignore the query, but this will result in a greater loss of context
    # we can also ignore the data, but this will result in a loss of data

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
        label_sequence = []

        query = obj.get("model_input")
        # the feature of this observation is the query and the response
        # keep separate variable for both components of the feature
        feature_query = ""
        feature_response = []
        # if the query is too long we may want to avoid using it in the features
        feature_query += f"{query} [SEP]"
        # keep a flag for cancellation in case of truncation overflow strategy
        break_loop = False
        # keep a flag for wrapping up in case of wrapping overflow strategy
        for sentence in obj["model_output_text_processed"]:
            for token in sentence:
                # add every lemma or original text as a feature
                if raw_input:
                    word_string = token.get("text")
                else:
                    word_string = token.get("lemma")
                if word_string is not None:
                    if include_POS:
                        raise Exception("POS-tags are not yet implemented")
                        # these 4 rows are optional: include POS-tags in the features
                        # also add a "ignore" label to the label_sequence
                        # upos = token.get("upos")
                        # xpos = token.get("xpos")
                    feature_response.append(word_string)
                    # for each word add also the label to the labels sequence
                    label = token.get("hallucination")
                    label_sequence.append(int(label) if label is not None else 0)

        if len(feature_query) + 1 + len(" ".join(feature_response)) <= max_length:
            feature = feature_query + " ".join(feature_response)
        # now if the feature_query + feature_response is too long, we need to apply an overflow strategy
        # Strategy 1: truncate the data
        # if we are overflowing
        else:
            # we can optionally skip the query if were overflowing
            if skip_overflowing_query:
                feature_query = "[SEP]"
            # we can skip an entire observation if it is overflowing
            if skip_overflowing_observation:
                continue
            # we can truncate the data to fit within max_length
            elif truncate_overflow:
                feature = feature_query
                # only add tokens to the response until the max_length is reached
                for i, token in enumerate(feature_response):
                    if len(feature) + 1 + len(token) < max_length:
                        feature += " " + token
                    else:
                        break
                # since we can only add until the ith token to the sequence before reaching max_length
                # we need to truncate the label_sequence as well
                label_sequence = label_sequence[:i]
            # we can split overflowing response into multiple observations with the same query head
            elif split_overflow:
                # make a new feature
                feature = feature_query
                # for iterating over all response tokens, keep track of the span of the feature (from i to j)
                i = 0
                # iterate over all tokens that need to go in a row
                for j, token in enumerate(feature_response):
                    # if the next token would make the feature too long,
                    # save the feature and make a new iteration with a new feature
                    if len(feature) + 1 + len(token) >= max_length:
                        # save this observation
                        features.append(feature)
                        labels.append(label_sequence[i:j + 1])
                        # reset the feature in case there is another iteration coming
                        feature = feature_query
                        # the current word is being skipped, if we dont append it to the feature now
                        # therefore add it to the feature, but truncate it to the max_length just in case
                        # TODO: this is a bit of a hack, maybe add a warning here to analyse if it happens and if its a problem
                        feature += " " + token
                        feature = feature[:max_length]
                    else:
                        # add the token to the feature
                        feature += " " + token
                    # set the left index to the next index in case there is another iteration coming
                    i = j + 1
                # save the last data in the lists
                features.append(feature)
                labels.append(label_sequence[i:j + 1])
                # skip the rest of the loop
                continue
        # save the data in the lists
        features.append(feature)
        labels.append(label_sequence)

    if len(features) != len(labels):
        raise Exception("The number of features and labels do not match!")

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
    if flatten_output:
        all_predictions = torch.cat(all_predictions, dim=0)  # Shape: (total_num_samples, seq_len)
    return all_predictions



def save_lists_to_delim_file(output_path, *args, delimiter="[DELIM]"):
    # *args are an unknown number of lists with the same length
    # write a csv file where the ith row is the ith element of each list
    with open(output_path, "w") as f:
        for i in range(len(args[0])):
            row = [str(arg[i]).replace("\n", "") for arg in args]
            f.write(f"{delimiter}".join(row) + "\n")

