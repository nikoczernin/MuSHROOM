import pandas as pd
import os
import json
from stanza.utils.conll import CoNLL


def read_original_data_json(data_path: str) -> dict:
    """
    Loads JSON and JSONL data files from subfolders within the specified data path,
    creating DataFrames for 'sample', 'train', and 'val' data.

    Parameters:
    - data_path (str): Path to the main directory containing subfolders with data files.

    Returns:
    - list: A list of DataFrames for 'sample', 'train', and 'val'.
    """

    # Dictionary to map folder names to empty DataFrames
    map_folder = {'sample': pd.DataFrame(),
                  'train': pd.DataFrame(),
                  'val': pd.DataFrame()}

    # Iterate over each folder name in the map
    for folder in map_folder.keys():
        # Define the path to each subfolder (sample, train, val)
        data_subfolder_path: str = os.path.join(data_path, folder)

        # Temporary list to store DataFrames loaded from each file in the subfolder
        df_list: list = []

        # Iterate over each file in the subfolder
        for filename in os.listdir(data_subfolder_path):
            # Check if the file is a .json or .jsonl file
            if filename.endswith(('.json', '.jsonl')):
                # Read the JSON/JSONL file into a DataFrame and append it to df_list
                df_list.append(pd.read_json(os.path.join(data_subfolder_path, filename), lines=True))

        # Concatenate all DataFrames from the folder into a single DataFrame
        map_folder[folder] = pd.concat(df_list)

    # Return a list of DataFrames for sample, train, and val
    return map_folder


def test_raw_data():
    # Define the path to the data directory (assumes it is one level up in '../data')
    DATA_DIR = os.getcwd() + '/../data'
    # Call the function to load data and store DataFrames in df_list
    df_list = read_original_data_json(DATA_DIR)
    # Print the list of DataFrames to verify the data has loaded correctly
    print(df_list)


def load_conll_data(path):
    docs = CoNLL.conll2doc(path)
    return docs


if __name__ == "__main__":
    path = "../data/preprocessed/sample_preprocessed.json"
    with open(path, "r") as f:
        data = json.load(f)
        for row in data:
            tokens = []
            tokens.append("[CLS]")
            for sentence in row["model_input_processed"]:
                for token in sentence:
                    try:tokens.append(token["lemma"])
                    except: pass
            for sentence in row["model_output_text_processed"]:
                for token in sentence:
                    try:tokens.append(token["lemma"])
                    except: pass
            print(len(tokens))
