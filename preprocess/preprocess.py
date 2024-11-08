import pandas as pd
import stanza
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from stanza.pipeline.multilingual import MultilingualPipeline
import os

# Download required Stanza language models for multilingual processing
stanza.download(lang="multilingual")  # Download the model for multilingual processing


class Preprocess:
    """
    A class for text preprocessing that includes methods for:
    - language model downloading and updating,
    - text processing using Stanza pipelines,
    - saving processed text in CoNLL format,
    - printing lemmatized text.
    """

    def __init__(self):
        """
        Initialize the Preprocess class with a dictionary of language pipelines.
        - A 'MultilingualPipeline' is created for automatic language detection.
        """
        # Keep a dictionary of language pipelines for processing multiple languages
        self.PIPELINES = {
            # Initialize with a multilingual pipeline to support multiple languages automatically
            "auto": MultilingualPipeline(processors='tokenize,lemma,pos')
        }


    def update_languages(self, langs):
        """
        Adds a language model for each specified language if it has not already been downloaded.

        Parameters:
        - langs (str or list): Language codes to add (e.g., 'en' for English).
        """
        # If a single language code is provided, convert it to a list for consistency
        if isinstance(langs, str): langs = [langs]

        # Add each language, ensuring no duplicates and downloading if needed
        for lang in list(set(langs)):
            lang = lang.lower()  # Stanza language codes are lowercase by convention
            if lang not in self.PIPELINES.keys():
                print(f"Downloading missing language: {lang}")
                stanza.download(lang)  # Download the language model for Stanza
                # Add a new pipeline for this language with specific processors
                self.PIPELINES[lang] = stanza.Pipeline(lang, processors='tokenize,lemma,pos')


    def preprocess(self, docs: list, lang="auto"):
        """
        Processes a list of text documents, converting each to a Stanza Document format.

        Parameters:
        - docs (list): List of strings (documents) to preprocess.
        - lang (str): Language code (default is "auto" for automatic detection).

        Returns:
        - List[Document]: List of processed Document objects.
        """
        # If a single document string is passed, convert it to a list
        if isinstance(docs, str): docs = [docs]

        # Convert each text document to a Stanza Document object with the text as input
        docs = [Document([], text=text) for text in docs]

        # Use the appropriate language pipeline, defaulting to "auto" if no language is specified
        if isinstance(lang, str):
            lang = lang.lower()
        else:
            lang = "auto"

        # Process the documents using the selected pipeline and return the processed text
        processed_text = self.PIPELINES[lang](docs)

        return processed_text


    @staticmethod
    def save_processed_text(processed_text, path: str, filename: str):
        """
        Saves the processed text in CoNLL format to the specified path.

        Parameters:
        - processed_text: The processed text (list of Stanza Document objects).
        - path (str): Directory path to save the .conll file.
        - filename (str): Name of the file to save.
        """

        # Ensure the directory exists; create it if it doesn't
        if not os.path.exists(path):
            os.makedirs(path)

        # Append ".conllu" extension if not already present
        if not filename.endswith(".conllu"):
            filename += ".conllu"

        # Save each processed document in CoNLL format
        print("Saving to", path)
        with open(f"{path}/{filename}", 'w', encoding='utf-8') as f:
            for doc in processed_text:
                CoNLL.write_doc2conll(doc, f)

        print("\tSaving successful!")


    @staticmethod
    def print(processed_text, max_lines=None):
        """
        Prints the lemmatized words in the processed text up to max_lines.

        Parameters:
        - processed_text: The processed text (list of Stanza Document objects).
        - max_lines (int or None): Maximum number of lines to print. Prints all lines if None.
        """

        # Iterate through processed documents and sentences to print lemmatized words
        for doc in processed_text:
            for sentence in doc.sentences:
                for word in sentence.words:
                    print(word.lemma, end=" ")  # Print each lemmatized word with a space separator
                print()  # New line after each sentence



# Placeholder functions for specific data processing tasks, which can be expanded as needed
def get_labeled_hallucination_by_index(df: pd.DataFrame) -> pd.DataFrame:
    pass


def get_ngrams_around_hallucinations() -> pd.DataFrame:
    pass


def test():
    """
    A test function to demonstrate the Preprocess class functionality.
    - Processes a list of sample text inputs
    - Prints lemmatized output and saves it in CoNLL format
    """

    # Sample text input for preprocessing
    text = [
        "Es ist die blanke Wahrheit: die FPÖ kann scheißen gehn!",
        "But if you ask me, the don't even deserve my hatred!",
        # "جليقة، التي تعرف الآن باسم كوريا الجنوبية، تتأ",
        # "सजायाफ्ता कैदियों को टेलीविज़न(टेलीविज़न) सीरीज़ 'जेल में बंद' के लिए वाइट कॉलर टीवी के रूप में पेश किया जाता है।...",
        "420 blaze it!"
    ]

    # Initialize the Preprocess object and update languages if necessary
    preprocessor = Preprocess()
    preprocessor.update_languages("en")  # Add English pipeline if needed

    # Process each line in the text to match the final intended processing method
    processed = [preprocessor.preprocess(t, "en")[0] for t in text]

    # Print the processed lemmatized output
    print(type(processed))
    Preprocess.print(processed)

    # Save the processed output in CoNLL format
    test_path = "../data/output/test"
    test_filename = "test.conllu"
    preprocessor.save_processed_text(processed, test_path, test_filename)

    # Load the saved CoNLL file to verify the saving process
    docs = CoNLL.conll2doc(f"{test_path}/{test_filename}")
    print(docs)
    print("Great success, I like!")


# Run the test function if the script is executed directly
if __name__ == "__main__":
    test()
