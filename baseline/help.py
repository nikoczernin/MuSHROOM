
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
