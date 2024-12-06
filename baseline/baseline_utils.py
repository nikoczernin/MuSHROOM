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
def get_data_for_training():
    # read the json of the preprocessed data
    with open('../data/output/preprocessing_outputs/sample_preprocessed.json') as f:
        sample = json.load(f)

    long_data = []
    for obj in sample:
        # now create a new object for each preprocessing mode output token and append it to the long_data list
        # we iterate over the processed token objects, with the iterating number being i
        # the ith token should correspond with the ith label
        # TODO: should we use the full original query or a concatenated version of its lemmas?
        query = obj.get("model_input")
        for sentence in obj["model_output_text_processed"]:
            for token in sentence:
                lemma = token.get("lemma")
                upos = token.get("upos")
                xpos = token.get("xpos")
                label = token.get("hallucination")
                long_data.append({
                    "query": query,
                    "lemma": lemma,
                    "upos": upos,
                    "xpos": xpos,
                    "label": int(label) if label is not None else 0
                })

    # [CLS] query [SEP] a single token from the answer [SEP] UPOS: the upos of the token, XPOS: the xpos of the token [SEP]
    features = [(f"[CLS] {obj.get('query')} "
                 f"[SEP] {obj.get('lemma')} "
                 f"[SEP] UPOS: {obj.get('upos')} "
                 f"[SEP] {obj.get('xpos')}") for obj in long_data]
    labels = [obj.get('label') for obj in long_data]

    if len(features) != len(labels):
        raise Exception("The number of features and labels do not match!")

    return features, labels
