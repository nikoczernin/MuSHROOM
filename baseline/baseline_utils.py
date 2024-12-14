import json
from functools import wraps
import time


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
def get_data_for_NN(datapath, include_POS=False, max_length=512,  ignore_label=-100,
                    truncate_overflow=True, skip_overflowing_query=False, skip_overflowing_observation=False,
                    split_overflow=False):
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
        feature_query += f"[CLS] {query} [SEP]"
        # keep a flag for cancellation in case of truncation overflow strategy
        break_loop = False
        # keep a flag for wrapping up in case of wrapping overflow strategy
        for sentence in obj["model_output_text_processed"]:
            for token in sentence:
                # add every lemma as a feature
                lemma = token.get("lemma")
                if lemma is not None:
                    if include_POS:
                        raise Exception("POS-tags are not yet implemented")
                        # these 4 rows are optional: include POS-tags in the features
                        # also add a "ignore" label to the label_sequence
                        # upos = token.get("upos")
                        # xpos = token.get("xpos")
                    feature_response.append(lemma)
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
                feature_query = "[CLS] [SEP]"
            # we can skip an entire observation if it is overflowing
            if skip_overflowing_observation: continue
            # we can truncate the data to fit within max_length
            elif truncate_overflow:
                feature = feature_query
                # only add tokens to the response until the max_length is reached
                for i, token in enumerate(feature_response):
                    if len(feature) + 1 + len(token) < max_length:
                        feature += token
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
                        labels.append(label_sequence[i:j+1])
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
                labels.append(label_sequence[i:j+1])
                # skip the rest of the loop
                continue
        # save the data in the lists
        features.append(feature)
        labels.append(label_sequence)

    if len(features) != len(labels):
        raise Exception("The number of features and labels do not match!")

    return features, labels


import torch
import numpy as np
import random


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
