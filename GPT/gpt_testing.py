import pandas as pd
import os
import openai
from stanza.utils.conll import CoNLL
import json
import ast
'''
def load_conll_data(path):
    file = CoNLL.conll2doc(path)
    docs=file.to_dict()
    return docs

input=load_conll_data(f"data/output/preprocessing_outputs/train_model_input.conllu")
output=load_conll_data(f"data/output/preprocessing_outputs/train_model_output_text.conllu")
'''
with open(r"D:\School\NLP\NLP_Test1_key.txt", 'r') as file:
    content = file.read()

client = openai.OpenAI(api_key=content)

#data=json.load(open(r"data\preprocessed\sample_preprocessed.json"))
data=[{"model_input":'Where is Austria located?',"model_output_text":'Australia is located in America.'},
        {"model_input":'When was Kaiser Franz Josef born?',"model_output_text":'He was born in 2020.'}]
spans=[]

for i in range(len(data)):
    completion = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "You are an assistant tasked with finding errors in responses."},
            {
                "role": "user",
                "content": f'The question was "{data[i]['model_input']}", and the response was "{data[i]['model_output_text']}". Identify the parts of the answer that are incorrect or unsupported by the question. Answer with only the incorrect spans in the answer as a list of token ranges (start and end indices), like in this example: "[(0,3),(8,20)]".'
            }
        ]
    )
    spans.append(ast.literal_eval(completion.choices[0].message.content))

for i in range(len(data)):
    print(f"\nQuestion: {data[i]['model_input']}\nAnswer: {data[i]['model_output_text']}\nHallucinated spans:")
    for j in spans[i]:
        print(data[i]['model_output_text'][j[0]:j[1]])
