import os

# Import function to read data from folders and preprocess class for text processing
from preprocess.load_data import read_data_from_folders
from preprocess.preprocess import Preprocess

# Import the lemmatizer module (change to use simplemma-based lemmatizer)
# TODO: is this obsolete now that we have a preprocess package?
from lemmatize.lemmatizer_simplemma import Lemmatizer

if __name__ == "__main__":
    # Define the path to the data directory
    DATA_DIR = os.getcwd() + '/data'

    # Load sample, train, and validation data from folders as a list of DataFrames
    df_list = read_data_from_folders(DATA_DIR)

    # TODO: this is the implementation of the lemmatization package
    # this may be obsolete
    # cols_to_lemmatize = ['model_input', 'model_output_text']
    # for df in df_list:
    #     print(f"Languages in this dataframe: {df.lang.unique().tolist()}")
    #     for col in cols_to_lemmatize:
    #         df[col] = df.apply(
    #             lambda row: Lemmatizer.lemmatize_text_input(
    #                 text=row[col],
    #                 lang=row['lang']
    #             ), axis=1
    #         )


    # Print the list of DataFrames to verify the data
    print(df_list)

    # Create an instance of the Preprocess class for text processing
    preprocessor = Preprocess()

    # Define columns to process and corresponding DataFrame names
    cols_to_process = ['model_input', 'model_output_text']
    df_names = ["sample", "train", "val"]

    # Loop over each DataFrame and column to preprocess text data
    for i, df_name in enumerate(df_names):
        for col in cols_to_process:
            print("Processing", df_list[i], "---", col)
            # Retrieve the text data for the specified column in the current DataFrame
            doc = df_list[i][col]

            # Process the document using the Preprocess class
            doc_processed = preprocessor.preprocess(doc)

            # Define a filename for saving the processed text
            foldername = f"{df_names[i]}_{col}_processed"

            # Save the processed text in CoNLL format to the specified path
            Preprocess.save_processed_text(doc_processed, f"data/output/{foldername}")
