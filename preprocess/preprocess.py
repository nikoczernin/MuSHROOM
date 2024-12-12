import pandas as pd
import stanza
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from stanza.pipeline.multilingual import MultilingualPipeline
import os
import json
from functools import wraps
import time
from pprint import pprint

# Download required Stanza language models for multilingual processing
from load_data import read_original_data_json

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
    def save_processed_text_conllu(processed_text, path: str, filename: str):
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
        #"सजायाफ्ता कैदियों को टेलीविज़न(टेलीविज़न) सीरीज़ 'जेल में बंद' के  है।...",
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
    preprocessor.save_processed_text_conllu(processed, test_path, test_filename)

    # Load the saved CoNLL file to verify the saving process
    docs = CoNLL.conll2doc(f"{test_path}/{test_filename}")
    print(docs)
    print("Great success, I like!")


def preprocess_project_milestone1(sample=True, train=True, val=True):
    # Define the path to the data directory
    # os.getcwd() returns the current working directory; adding '/data' to it specifies the data folder
    DATA_DIR = os.getcwd() + '/../data'

    # Load data from 'sample', 'train', and 'validation' folders as a dictionary
    # The dictionary keys are the folder names, and values are the DataFrames with the data
    dataDict = read_original_data_json(DATA_DIR)

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
            output_foldername = f"preprocessing_outputs"

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
            Preprocess.save_processed_text_conllu(processed_data, f"data/output/{output_foldername}", f"{df_name}_{col}.conllu")


def preprocess(data, data_name, output_folder="../data/"):
    # Create an instance of the Preprocess class to handle text processing operations
    preprocessor = Preprocess()

    # define a list of all observations, each being a dictionary
    # we then save each list of observations as a json file
    all_observations = []


    print("Processing", data_name)

    # Update the preprocessor to include languages in 'langs' (download models if missing)
    langs = data["lang"]
    print(f"All detected languages in this dataset: {list(set(data['lang']))}")
    preprocessor.update_languages(langs)

    # Iterate over each row to process text with language-specific handling
    # This is especially necessary because some languages (e.g., Hindi) might require special handling
    for row in data.itertuples():
        # itertuples() yields rows as row objects
        # get the attributes from the row object
        i = getattr(row, "Index")
        model_input = getattr(row, "model_input").strip()
        model_output_text = getattr(row, "model_output_text").strip()
        lang = getattr(row, "lang")
        try:
            hard_labels = getattr(row, "hard_labels")
        except AttributeError:
            hard_labels = None

        # Process the text using the Preprocess class, which might unpack it if necessary
        # This processing is row-by-row because some languages could trip up a bulk pipeline
        # print(f"\tProcessing input: ({lang}) \"{model_input[:100]} ...\"")
        model_input_processed = preprocessor.preprocess(model_input, lang)[0].to_dict()
        # print(f"\tProcessing output: ({lang}) \"{model_output_text[:100000]} ...\"")
        model_output_text_processed = preprocessor.preprocess(model_output_text, lang)[0].to_dict()

        # the hard labels correspond with character indices of the model_output_text
        # the Stanza preprocessing saves the original word starting and ending character indices
        # these we can compare with the character index hard labels to check if it is a hallucination
        # we do that here to save the hard label of each token in the preprocessed data
        if hard_labels is not None:
            for sentence in model_output_text_processed:
                for token in sentence:
                    # not all token objects have the start_char and end_char attributes, the others you can skip
                    if "start_char" in token.keys():
                        print(hard_labels)
                        for left, right in hard_labels:
                            if left <= token["start_char"] <= right:
                                token["hallucination"] = True
                                break
                            else:
                                token["hallucination"] = False


        # Append the processed data and all wanted attributes to our list for later saving
        all_observations.append({
            # original data
            "model_input": model_input,
            "model_output_text": model_output_text,
            # processed data
            "model_input_processed": model_input_processed,
            "model_output_text_processed": model_output_text_processed,
            "lang": lang,
            "hard_labels": hard_labels,
        })
    # create it in advance otherwise the prep well go stoopid
    # After processing all rows in the column, save the processed text in JSON format
    print(f"Saving preprocessed data to '{output_folder}/{data_name}_preprocessed.json'")
    with open(f"{output_folder}/{data_name}_preprocessed.json", "w") as f:
        json.dump(all_observations, f)


def preprocess_project(sample=True, train=True, val=True, output_folder="../data/"):
    # Define the path to the data directory
    # os.getcwd() returns the current working directory; adding '/data' to it specifies the data folder
    DATA_DIR = os.getcwd() + '/../data'

    # Load data from 'sample', 'train', and 'validation' folders as a dictionary
    # The dictionary keys are the folder names, and values are the DataFrames with the data
    dataDict = read_original_data_json(DATA_DIR)

    dataframes_to_process = []
    if sample: dataframes_to_process.append("sample")
    if val: dataframes_to_process.append("val")
    if train: dataframes_to_process.append("train")

    # Loop over each DataFrame and each row to preprocess text data
    for df_name, df in dataDict.items():
        if df_name in dataframes_to_process:
            preprocess(df, df_name, output_folder)



# Run the test function if the script is executed directly
if __name__ == "__main__":
    # test()
    # preprocess_project(sample=False, train=False, val=True, output_folder="../data/preprocessed")
    # print("Great success, I like!")
    preprocess(pd.read_json("../data/exploration/exploration.json", lines=True), "exploration", output_folder="../data/preprocessed")

