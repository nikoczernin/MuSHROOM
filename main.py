import os

from preprocess.load_data import read_data_from_folders

if __name__ == "__main__":
    DATA_DIR = os.getcwd() + '\\data'
    sample_data, train_data, val_data = read_data_from_folders(DATA_DIR)

