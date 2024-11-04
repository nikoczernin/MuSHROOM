import pandas as pd
import stanza
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from stanza.pipeline.multilingual import MultilingualPipeline
import os

# Download required Stanza language models
stanza.download(lang="multilingual")  # Download model for multilingual processing


class Preprocess:
    # Class attribute to track downloaded languages
    LANGUAGES = []

    def __init__(self):
        # Initialize with a MultilingualPipeline for processing multiple languages
        self.Pipeline = Preprocess.get_pipeline()

    @staticmethod
    def add_language(lang):
        """
        Adds a specified language model if not already downloaded,
        and updates the LANGUAGES list.

        Parameters:
        - lang (str): Language code (e.g., 'en' for English)
        """
        if lang not in Preprocess.LANGUAGES:
            stanza.download(lang)
            Preprocess.LANGUAGES.append(lang)

    @staticmethod
    def get_pipeline():
        """
        Returns a multilingual pipeline with tokenization, lemmatization, and POS tagging processors.

        Returns:
        - MultilingualPipeline: A Stanza pipeline configured for multiple languages.
        """
        return MultilingualPipeline(processors='tokenize,lemma,pos')

    def preprocess(self, docs: list, lang="en"):
        """
        Processes a list of text documents, converting each to a Stanza Document format.

        Parameters:
        - docs (list): List of strings (documents) to preprocess.
        - lang (str): Language code (default is "en" for English).

        Returns:
        - List[Document]: List of processed Document objects.
        """
        # Convert each input text to a Stanza Document object
        docs = [Document([], text=text) for text in docs]
        # Return the processed documents using the initialized pipeline
        return self.Pipeline(docs)

    @staticmethod
    def save_processed_text(processed_text, path: str):
        """
        Saves the processed text in CoNLL format to the specified path.

        Parameters:
        - processed_text: The processed text (list of Stanza Document objects).
        - path (str): Path to save the .conll file.
        """

        # Create the directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)


        # Save the processed text in CoNLL format
        print("Saving to", path)
        for i,doc in enumerate(processed_text):
            CoNLL.write_doc2conll(doc, f"{path}/{i}.conllu")

    @staticmethod
    def print(processed_text, max_lines=None):
        """
        Prints the lemmatized words in the processed text up to max_lines.

        Parameters:
        - processed_text: The processed text (list of Stanza Document objects).
        - max_lines (int or None): Maximum number of lines to print. Prints all lines if None.
        """

        # Print lemmatized words, observing the line limit
        for doc in processed_text:
            for sentence in doc.sentences:
                for word in sentence.words:
                    print(word.lemma, end=" ")
                print()


# Placeholder functions for specific data processing tasks
def get_labeled_hallucination_by_index(df: pd.DataFrame) -> pd.DataFrame:
    pass


def get_ngrams_around_hallucinations() -> pd.DataFrame:
    pass


if __name__ == "__main__":
    # Sample text input for preprocessing
    text = [
        "Es ist die blanke Wahrheit: die FPÖ kann scheißen gehn!",
        # "But if you ask me, the don't even deserve my hatred!",
        "420 blaze it!"
    ]

    # Initialize the Preprocess object and process the text
    preprocessor = Preprocess()
    processed = preprocessor.preprocess(text)

    # Print processed output
    print(type(processed))
    Preprocess.print(processed)

    # Save processed output in CoNLL format
    preprocessor.save_processed_text(processed, "../data/output/test_en")

    # read in the saved data to make sure it works
    read_path = "../data/output/test_en"
    docs = [CoNLL.conll2doc(f"{read_path}/{path}") for path in os.listdir(read_path)]
    print(docs)
