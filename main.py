import os

# Import function to read data from folders and preprocess class for text processing
from preprocess.load_data import read_data_from_folders
from preprocess.preprocess import Preprocess

# Import the lemmatizer module (change to use simplemma-based lemmatizer)
# TODO: is this obsolete now that we have a preprocess package?
from lemmatize.lemmatizer_simplemma import Lemmatizer

if __name__ == "__main__":
    # debugging parameter
    # when debugging, go row for row in preprocessing and print every row
    DEBUG = True
    if DEBUG: print("######## Debuggin mode is on!! ########")




    # Define the path to the data directory
    DATA_DIR = os.getcwd() + '/data'

    # Load sample, train, and validation data from folders as a dictionary
    dataDict = read_data_from_folders(DATA_DIR)

    # TODO: val - model_output_text fails with a error
    # TypeError: argument of  type 'NoneType' is not iterable
    # Investigate!

    val = dataDict["val"]
    print(val.model_output_text)





    # Create an instance of the Preprocess class for text processing
    preprocessor = Preprocess()

    # Define columns to process and corresponding DataFrame names
    cols_to_process = ['model_input', 'model_output_text']
    # cols_to_process = ['model_output_text']
    df_names = ["sample", "train", "val"]





    # Loop over each DataFrame and column to preprocess text data
    for df_name, df in dataDict.items():
        if df_name != "val": continue # TODO: delete this before pushing!

        for col in cols_to_process:
            # skip if the current df doesnt contain the current col
            if col not in df.columns:
                print(f"The dataframe {df_name} does not contain the column {col}")
                continue

            print("Processing", df_name, "---", col)
            # Retrieve the text data for the specified column in the current DataFrame
            doc = df[col]

            # Define a filename for saving the processed text
            foldername = f"{df_name}_{col}_processed"

            if DEBUG:
                for i, text in enumerate(df[col]):
                    if i <= 132: continue
                    print(f"Processing: {text} -->")
                    # Process the document using the Preprocess class
                    text_processed = preprocessor.preprocess(text)
                    print(text_processed)
                    # Save the processed text in CoNLL format to the specified path
                    Preprocess.save_processed_text(text_processed, f"data/output/{foldername}_DEBUG")

            elif not DEBUG:
                # Process the document using the Preprocess class
                doc_processed = preprocessor.preprocess(doc)
                # Save the processed text in CoNLL format to the specified path
                Preprocess.save_processed_text(doc_processed, f"data/output/{foldername}")
