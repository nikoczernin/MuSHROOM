import os

from preprocess.load_data import read_data_from_folders
# from lemmatize.lemmatizer import Lemmatizer
from lemmatize.lemmatizer_simplemma import Lemmatizer

if __name__ == "__main__":
    DATA_DIR = os.getcwd() + '/data'
    df_list = read_data_from_folders(DATA_DIR)

    cols_to_lemmatize = ['model_input', 'model_output_text']

    #for df in df_list:
    for df in [df_list[0]]:
        for col in cols_to_lemmatize:
            df[col] = df.apply(
                lambda row: Lemmatizer.lemmatize_text_input(
                    text=row[col],
                    lang=row['lang'].upper()
                ), axis=1
            )

    print("ready")

