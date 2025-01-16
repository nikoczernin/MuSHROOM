import pandas as pd
import os
import openai
# from stanza.utils.conll import CoNLL
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
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# data=json.load(open(r"data\val\mushroom.en-val.v2.jsonl"))
# data=[{"model_input":'Where is Austria located?',"model_output_text":'Australia is located in America.'},
#         {"model_input":'When was Kaiser Franz Josef born?',"model_output_text":'He was born in 2020.'}]
# open jsonl file
lang = ["ar", "fi","fr", "hi", "it", "sv", "zh"]

for lang in lang:
    data = []
    with open(f"data/val/mushroom.{lang}-val.v2.jsonl") as f:
        for line in f:        
            data.append(json.loads(line))

    spans=[]

    for i in range(len(data)):
        completion = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are an assistant tasked with finding errors in responses."},
                {
                    "role": "user",
                    "content": (
                        f'The question was "{data[i]["model_input"]}", and the response was "{
                            data[i]["model_output_text"]}". '
                        "Analyze the model's response to detect factual inaccuracies or hallucinations. For the given input and output pair, return a JSON object in the following format:"
                        '{'
                        '  "hallucinated_span": string,'
                        '  "indices": [[start_index, end_index],[start_index, end_index], ...]'
                        '}'
                        "If no hallucination is detected, return:"
                        '{'
                        '  "hallucinated_span": null,'
                        '  "indices": null'
                        '}'
                        "Ensure the response contains only valid JSON, with double quotes and no extra text or newlines."
                    )
                }
            ]
        )

        try:
            print(f"\nQuestion: {data[i]['model_input']}\nAnswer: {data[i]['model_output_text']}\nHallucinated spans:")
            print(completion.choices[0].message.content)
            #vali
            response_content = completion.choices[0].message.content
            spans.append(json.loads(response_content))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Offending content: {response_content}")
            spans.append({"hallucinated_span": None, "indices": None})  # Fallback
        
        #write to file
        with open(f"data/val/gpt/mushroom.{lang}-val.v2.spans.jsonl", "a") as f:
            f.write(json.dumps(spans[i]) + "\n")