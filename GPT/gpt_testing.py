import pandas as pd
import os
import openai
from stanza.utils.conll import CoNLL
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

input=['Where is Austria located?']
output=['Austria is located in America.']

for i in range(len(input)):
    completion = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "You are an assistant tasked with finding errors in responses."},
            {
                "role": "user",
                "content": f'The question was "{input[i]}", and the response was "{output[i]}". Identify the parts of the answer that are incorrect or unsupported by the question. Output the hallucinated spans as a list of token ranges (start and end indices).'
            }
        ]
    )
    print(completion.choices[0].message)
