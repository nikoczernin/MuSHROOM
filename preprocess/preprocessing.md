# Preprocessing

## Introduction & data overview
Here we outline how we approached the preparation of the data for the task. 
There are 3 datasets for training, validation and a very small one as a sample. 
Since the labels are essentially just a form of indexing of the model output, 
the only free-text columns that need preprocessing are the model input question `model_input` 
and the model output `model_output_text`.  

## Approach & workflow
We prepared a preprocessing module `reprocess` with a class `Preprocess`, 
which handles most of the operative steps.  

We started out using the Spacy NLP library, which includes tools for 
tokenization and lemmatization. 
Out initial approach was patchworking a pipeline ourselves. 
At this point, we discovered how few NLP libraries 
support languages like Hindi, Mandarin and Arabic.  
For that reason we resorted to using the Stanza NLP library, which conveniently supports all languages included in 
the datasets. 

Stanza includes prebuilt ``Pipelines``, for which, during their initialization, the types of wanted processing steps 
can be specified (we went with tokenization, lemmatization and part-of-speach tagging). 
Upon execution of the processing, a single string or a series of strings are given as arguments, as well as a language 
code. Any language requires the download of a language specific model before processing, or the use of custom weights. 

Stanza has a Pipeline called `MultilingualPipeline`, which automatically detects even multiple different 
languages within a string and processes the substrings accordingly and only required a single, 
multilingual model download beforehand. The obvious approach was to use this Pipeline on the whole dataset, but it 
kept failing on some texts contained in the validation dataset. These texts were in Hindi, 
a language which should natively be supported, but processing it kept raising NoneType-Errors. 
This hurdle stopped us from using our initial approach of processing the whole Series of strings in one go 
and thus we went with another approach: We gathered all languages used in the dataset, which was specified for 
each row in the datasets, and download all required models. This slows down the first couple of 
observations, as the downloads take some time and disk space. 
We created a Python dictionary with Pipelines for each language. 
From then on, we process each line individually, using the corresponding language Pipeline respectively. 
This row-by-row approach was noticeably slower than the holistic approach of processing the whole dataset with a 
single Pipeline, but far easier to debug and apply individual fixes.  

Saving the processed data in the required standard format initially caused some trouble. 
We started out having to save eaceh row as a single ``conllu`` file, but after analysing the ``Document`` class in the 
Stanza library, we overcame this obstacle. 

## Testing
To test out the preprocessing on custom text, use the function ``test()`` in ``preprocess.preprocessing.py``. 
Add some lines of text to the `text` list and the desired languages, then run the function. 
It will download the specified languages (repeated execution will skip redundant downloads) 
and process the texts, then save them and re-read them using the ```preprocess.load_data``` module. 
You can set a language to `"auto"` to auto-detect the language using `MultilingualPipeline`.   

## Reproducing
To run the whole preprocessing, run the function ``preprocess.preprocess_project()``. 
It will iterate over the 3 datasets and the two string columns that need processing. 
It will then download all required language models (again, it skips the lengthy downloads after the first time), 
and process each row, finally saving the processed data in .conllu files, one file for each column.  
For example, the model output for the training data will have the following path: 
`./data/output/preprocessing_outputs/train_model_output_text.conllu`  

Again, you can use ```preprocess.load_data.test_conll_data(path:str)``` to load and inspect the output.

## Output of the processing
The process data is stored in a single ``conllu`` file, which, when read in contains Stanza's ``Document`` object. Within this document are stored the ``Sentence``, ``Entity``, and other similar components of the text in a format that we understand to be essentially a list of dictionaries.
We found two noteworthy tendencies when analyzing our final outputs. Firstly, when preprocessing contractions, for example "it's", the tool tokenizes it as separate "it" and "'s" tokens, which are then lemmatized as "it" and "be". However, it also leaves in the original "it's" in the token list, it simply does not lemmatize it further. Additionally, the contraction does not get its own ID, instead using a combination of the IDs of its separate parts i.e., if the ID of "it" and "be" are 3 and 4 respectively, then the ID of "it's" would be 3-4. This is unlikely to cause any issues, but we felt it was an important curiosity to highlight.
On the other hand, the second issue of note could cause some challenges. When dealing with a numbered list, sentences are broken up based on periods in a manner that does not reflect how we would intuitively consider breaking it up. Below is an example list:
"
1. First sentence.
2. Second sentence.
3. Third sentence.
"
This list would be considered six sentences by the tool, with "1.", "2.", and "3." each being their own sentence, along with the text in the list. It might be necessary to handle this issue in the future, depending on how this impacts our further usage, but we would first like to see how the rest of our process interacts with this before making modifications.

## References

1. Qi, Peng, Zhang, Yuhao, Zhang, Yuhui, Bolton, Jason, & Manning, Christopher D. 
   **Stanza: A Python Natural Language Processing Toolkit for Many Human Languages**. 
   *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations*, 2020.

2. Honnibal, Matthew, Montani, Ines, Van Landeghem, Sofie, & Boyd, Adriane.  
   **spaCy: Industrial-strength Natural Language Processing in Python**. 
   2020. doi: [10.5281/zenodo.1212303](https://doi.org/10.5281/zenodo.1212303)
