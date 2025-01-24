import json
from functools import wraps
import time
import torch
import numpy as np
import random

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



# Helper class for all hyperparameters
class NN_Args:
    def __init__(self):
        self.MAX_LENGTH = 128 * 4  # Max token length for mBERT
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

        self.include_query = True

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
        self.skip_overflowing_observation = False


def timer(func):
    """
    A decorator to measure and print the execution time of a function.

    NN_Args:
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
                    include_query=True,
                    skip_overflowing_observation=False,
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
        if include_query:
            # add the query tokens and labels
            for sentence in obj["model_input_processed"]:
                for token in sentence:
                    # add every lemma or original text as a feature
                    if raw_input: word_string = token.get("text")
                    else: word_string = token.get("lemma")
                    if word_string is not None:
                        # for each word add its string and the label to the sequences
                        feature_seq.append(word_string)
                        label_seq.append(-100)
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
                # for each word add its string and the label to the sequences
                feature_seq.append(word_string)
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

        # check if the length of the feature sequence is too long
        # if wanted, skip this observation to avoid observations with missing context
        if skip_overflowing_observation:
            if len(feature_seq) > max_length:
                continue

        if len(feature_seq) != len(label_seq):
            raise Exception(f"The number of features and labels do not match in sample {sample_nr}!")

        # save the data in the lists (Nones are not allowed here)
        features.append([x if x is not None else "[PAD]" for x in feature_seq])
        labels.append([y if y is not None else -100 for y in label_seq])

    if len(features) != len(labels):
        raise Exception("The number of features and labels do not match!")

    # join the features into a single string
    # Edit: I stopped doing it here, because joining and splitting does not
    # result in the same sequence of tokens as before!!!
    # features = [" ".join(feature_seq) for feature_seq in features]
    print(f"Data prepared, there are {len(features)} observations in the data.")
    return features, labels


def set_seed(seed: int):
    """
    Sets the seed for reproducibility across multiple libraries.
    NN_Args:
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


def evaluate_predictions(y, yhat, labels_ignore=[-100], accuracy=True, recall=True, precision=False, f1=False):
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
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for i in range(len(y)):
        y[i] = np.array(y[i])
        yhat[i] = np.array(yhat[i])
        # remove items where y == -100
        valid_indices = np.isin(y[i], labels_ignore, invert=True)
        y[i] = y[i][valid_indices]
        yhat[i] = yhat[i][valid_indices]
        # get evaluation metrics
        accuracies.append(accuracy_score(y[i], yhat[i]))
        precisions.append(precision_score(y[i], yhat[i], average="binary"))
        recalls.append(recall_score(y[i], yhat[i], average="binary"))
        f1s.append(f1_score(y[i], yhat[i], average="binary"))

    # compute the average metrics
    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1s)

    # also get matrics for all data concatenated
    y_flat = np.concatenate(y).flatten()
    yhat_flat = np.concatenate(yhat).flatten()

    # Compute flattened metrics
    precision = precision_score(y_flat, yhat_flat, average="binary")
    recall = recall_score(y_flat, yhat_flat, average="binary")
    f1 = f1_score(y_flat, yhat_flat, average="binary")
    accuracy = accuracy_score(y_flat, yhat_flat)

    metrics = {}
    if accuracy:
        metrics["Accuracy"] = mean_accuracy
        metrics["Accuracy_flat"] = accuracy
    if precision:
        metrics["Precision"] = mean_precision
        metrics["Precision_flat"] = precision
    if recall:
        metrics["Recall"] = mean_recall
        metrics["Recall_flat"] = recall
    if f1:
        metrics["F1"] = mean_f1
        metrics["F1_flat"] = f1
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

