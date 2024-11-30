from pprint import pprint
import pandas as pd
from load_data import load_conll_data

import json
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import os


print("hi")

# # read sample data
# sample = pd.read_json('../data/sample/sample_set.v1.json', lines=True)
# # print(sample.iloc[2])


# load the preprocessed sample data from JSON
with open('../data/output/preprocessing_outputs/sample_preprocessed.json') as f:
    samplep = json.load(f)

# pprint(samplep[2])
# df = pandas.read_json([samplep])


def get_lemmas_from_stanza_list(stanza_list, split_ngrams=True, skip_punct=False):
    """
    Extracts lemmas from a Stanza-processed list of sentences, with optional handling of n-grams and punctuation.
    Parameters:
    - stanza_list (list): A list of sentences, where each sentence is a list of word dictionaries.
                          Each word dictionary includes attributes like 'id', 'lemma', 'text', and 'upos'.
                          Example: [{'id': 1, 'lemma': 'example', 'text': 'example', 'upos': 'NOUN'}, ...].
    - split_ngrams (bool): Determines how to handle n-grams:
                           - If True: Keeps only the split components of n-grams (e.g., "don't" becomes ["do", "n't"]).
                           - If False: Skips split components and retains only the joint n-gram (e.g., "don't").
    - skip_punct (bool): If True, skips tokens labeled as punctuation (e.g., ".", ",", "!", etc.).
    Returns:
    - list: A list of lemmas extracted from the input sentences, optionally handling n-grams and skipping punctuation.
    Example:
    Input:
        stanza_list = [
            [{'id': 1, 'lemma': 'I', 'text': 'I', 'upos': 'PRON'},
             {'id': [2, 3], 'lemma': None, 'text': "don't", 'upos': 'VERB'},
             {'id': 2, 'lemma': 'do', 'text': 'do', 'upos': 'AUX'},
             {'id': 3, 'lemma': "n't", 'text': "n't", 'upos': 'PART'}],
            [{'id': 1, 'lemma': 'run', 'text': 'run', 'upos': 'VERB'},
             {'id': 2, 'lemma': '.', 'text': '.', 'upos': 'PUNCT'}]
        ]
    Output (split_ngrams=True, skip_punct=True):
        ['I', 'do', "n't", 'run']
    Output (split_ngrams=False, skip_punct=False):
        ['I', "don't", 'run', '.']
    """
    lemmas = []  # List to store extracted lemmas
    ids_to_skip = []  # List to track indices of words to skip

    # Iterate over each sentence (outer lists)
    for sentence in stanza_list:
        # Iterate over each word (inner lists), represented as dictionary objects
        for word in sentence:
            # Extract the word's index
            word_index = word['id']

            # Handle n-grams (e.g., "don't" split into "do" and "n't")
            if isinstance(word_index, list):
                if not split_ngrams:
                    # If keeping joint n-grams, skip split components and store the joint word's text
                    ids_to_skip.extend(word_index)  # Mark split components for skipping
                    lemmas.append(word['text'])  # Use the full joint word text
                continue  # Move to the next word

            # Skip punctuation if skip_punct is True
            if skip_punct and word['upos'] == 'PUNCT':
                continue

            # Skip words whose indices are marked in ids_to_skip
            if word_index in ids_to_skip:
                continue

            # Add the lemma of the word to the list
            lemmas.append(word['lemma'])

    return lemmas


# TODO: find out if the lemma tokens returned form our processed data and the function above are the same
#  as in the original sample data

sandor = samplep[0]
sandor_labels = sandor["hard_labels"]
print(sandor_labels)

# TODO: the issue lies within the hard labels, they are character indices of the original text
# not word indices
# we need to convert them to word indices

# get both texts
sandor_lemmas = get_lemmas_from_stanza_list(sandor["model_output_text_processed"], skip_punct=False, split_ngrams=True)
sandor_original = sandor["model_output_text"]
print("Lemmas:", sandor_lemmas)
print("Original:", sandor_original)
print("length original:", len(sandor_original))

# now for each sequence of hard labels, print the corresponding texts of both datasets
for label_pair in sandor_labels:
    start = label_pair[0]
    end = label_pair[1]
    print(" ".join(sandor_lemmas[start:end+1]))
    print(sandor["model_output_text"][start:end+1])
    print("####")