import numpy as np
from pprint import pprint
import json
import pandas as pd

demo_loc = "data/sample/sample_set.v1.json"
with open(demo_loc, "r") as f:
    samp = f.readlines()
samp = [json.loads(l) for l in samp]
pprint(samp[0])
df = pd.DataFrame.from_dict(samp)
