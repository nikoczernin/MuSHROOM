import numpy as np
from pprint import pprint
import json
import pandas as pd
import os

# demo_loc = "data/sample/sample_set.v1.json"
# with open(demo_loc, "r") as f:
#     samp = f.readlines()
# samp = [json.loads(l) for l in samp]
# pprint(samp[0])
# df = pd.DataFrame.from_dict(samp)
# print(df)



# get all languages used in the training data
languages = []
# iterate through all files as load each jsonl file
train_loc = "data/train"
for filename in os.listdir(train_loc):
    if filename.endswith(".jsonl"):
        file_path = os.path.join(train_loc, filename)

        with open(file_path, "r") as f:
            train = f.readlines()
            train = [json.loads(l) for l in train]
            df_train = pd.DataFrame.from_dict(train)
            languages += df_train.lang.unique().tolist()

print(languages)
