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
def get_data_for_NN(datapath, include_POS=False, truncate_overflow=False, ignore_label=-100):
    # for each observation in the data, we want to create two training objects, saved in separate lists
    # 1: the features
    # 2: the labels
    # the features are the query and the response, prepended by a [CLS] token and separated by a [SEP] token
    # the labels are a list of binary labels (1 for hallucatinations)

    # TODO: often the data is too long to make for a reasonable input to the model
    # we can truncate the data to a maximum length, but this will result in a loss of information
    # we can also split the data in multiple observations, but this will result in a loss of context


    # read the json of the preprocessed data
    with open(datapath) as f:
        sample = json.load(f)

    features = []
    labels = []
    for obj in sample:
        # now create a new object for each preprocessing mode output token and append it to the long_data list
        # we iterate over the processed token objects, with the iterating number being i
        # the ith token should correspond with the ith label
        query = obj.get("model_input")
        # the feature of this observation is the query and the response
        feature = ""
        feature += f"[CLS] {query} [SEP]"
        # the label of this response is a list of binary labels (1 for hallucatinations)
        label_sequence = []
        for sentence in obj["model_output_text_processed"]:
            for token in sentence:
                feature += " " # add a single whitespace for every new token
                if include_POS:
                    # these 4 rows are optional: include POS-tags in the features
                    # also add a "ignore" label to the label_sequence
                    upos = token.get("upos")
                    feature += f" {upos}"
                    label_sequence.append(ignore_label)
                    xpos = token.get("xpos")
                    feature += f" {xpos}"
                    label_sequence.append(ignore_label)

                # add every lemma as a feature
                lemma = token.get("lemma")
                if lemma is not None:
                    feature += lemma + "" if lemma is not None else ""
                    # for each word add also the label to the labels sequence
                    label = token.get("hallucination")
                    label_sequence.append(int(label) if label is not None else 0)
        # save the data in the lists
        features.append(feature)
        labels.append(label_sequence)

    if len(features) != len(labels):
        raise Exception("The number of features and labels do not match!")

    return features, labels
