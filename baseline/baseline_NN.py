from pprint import pprint
import pandas as pd
from load_data import load_conll_data
import json

print("hi")
# read sample data
sample = pd.read_json('../data/sample/sample_set.v1.json', lines=True)
# print(sample.iloc[2])
# load the preprocessed sample data from JSON
with open('../data/output/preprocessing_outputs/sample_preprocessed.json') as f:
    samplep = json.load(f)

for line in samplep:
    print(line["model_output_text"])