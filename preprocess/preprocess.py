import pandas as pd
import stanza
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from stanza.pipeline.multilingual import MultilingualPipeline
import os

# Download required Stanza language models for multilingual processing
from preprocess.load_data import read_data_from_folders

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



def test():
    """
    A test function to demonstrate the Preprocess class functionality.
    - Processes a list of sample text inputs
    - Prints lemmatized output and saves it in CoNLL format
    """

    # Sample text input for preprocessing
    text = [
        "Es ist die blanke Wahrheit!",
        "But if you ask me, the don't even deserve my hatred!",
        "सजायाफ्ता कैदियों को टेलीविज़न(टेलीविज़न) सीरीज़ 'जेल में बंद' के  है।...",
    ]

    # Specify the languages of the sample texts
    # use "auto" for automatic language detection (error prone)
    languages = [
        "de",
        "en",
        "hi"
    ]

    while len(languages) < len(text):
        languages.append("auto")

    # Initialize the Preprocess object and update languages if necessary
    preprocessor = Preprocess()
    for lang in languages:
        preprocessor.update_languages(lang)

    # Process each line in the text to match the final intended processing method
    processed = [preprocessor.preprocess(t, languages[i])[0] for i, t in enumerate(text)]

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


def preprocess_project():
    # Define the path to the data directory
    # os.getcwd() returns the current working directory; adding '/data' to it specifies the data folder
    DATA_DIR = os.getcwd() + '/data'

    # Load data from 'sample', 'train', and 'validation' folders as a dictionary
    # The dictionary keys are the folder names, and values are the DataFrames with the data
    dataDict = read_data_from_folders(DATA_DIR)

    # Create an instance of the Preprocess class to handle text processing operations
    preprocessor = Preprocess()

    # Define the columns we want to process and the dataset names we're working with
    # Each column in cols_to_process will be processed in each DataFrame in dataDict
    cols_to_process = ['model_input', 'model_output_text']
    df_names = ["sample", "train", "val"]

    # Loop over each DataFrame and each specified column to preprocess text data
    for df_name, df in dataDict.items():
        for col in cols_to_process:
            # Check if the DataFrame contains the specified column; if not, skip it
            if col not in df.columns:
                print(f"The DataFrame '{df_name}' does not contain the column '{col}'")
                continue

            print("Processing", df_name, "---", col)

            # Retrieve the text data and languages for the specified column in the current DataFrame
            doc = df[col]
            langs = df["lang"]
            print(f"All detected languages in this dataset: {list(set(df['lang']))}")

            # Update the preprocessor to include languages in 'langs' (download models if missing)
            preprocessor.update_languages(langs)

            # Define the folder name for saving processed text output
            foldername = f"preprocessing_outputs"

            # Initialize an empty list to store processed data for this column
            processed_data = []

            # Iterate over each row to process text with language-specific handling
            # This is especially necessary because some languages (e.g., Hindi) might require special handling
            for i, text in enumerate(df[col]):
                # Retrieve the language for this row from the 'lang' column
                lang = langs.iloc[i]

                # Display debug information about the row being processed (first 100 characters of text)
                print(f"\tProcessing: ({lang}) \"{text[:100]} ...\"")

                # Process the text using the Preprocess class, which might unpack it if necessary
                # This processing is row-by-row because some languages could trip up a bulk pipeline
                text_processed = preprocessor.preprocess(text, lang)[0]

                # Append the processed text to our list for later saving
                processed_data.append(text_processed)

            # After processing all rows in the column, save the processed text in CoNLL format
            # Each dataset and column is saved as a separate file in <foldername>
            Preprocess.save_processed_text(processed_data, f"data/output/{foldername}", f"{df_name}_{col}.conllu")


# Run the test function if the script is executed directly
if __name__ == "__main__":
    test()
    # preprocess_project()
