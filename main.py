import os

# Import function to read data from folders and the Preprocess class for text processing
from preprocess.load_data import read_data_from_folders
from preprocess.preprocess import Preprocess





if __name__ == "__main__":

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
            foldername = f"holistic_outputs"

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