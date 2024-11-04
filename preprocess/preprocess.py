import pandas as pd
import stanza
from stanza.utils.conll import CoNLL

import os


class Preprocess:
    LANGUAGES = []

    @staticmethod
    def add_language(lang):
        if lang not in Preprocess.LANGUAGES:
            stanza.download(lang)
            Preprocess.LANGUAGES.append(lang)

    @staticmethod
    def get_pipeline(lang):
        if lang not in Preprocess.LANGUAGES:
            Preprocess.add_language(lang)
        return stanza.Pipeline(lang, processors='tokenize,lemma,pos')

    @staticmethod
    def preprocess(text, lang="en"):
        # returns a tokennized corpus in JSON format
        pipeline = Preprocess.get_pipeline(lang)
        text_processed = pipeline(text)
        return text_processed

    @staticmethod
    def save_processed_text(processed_text:str, path:str):
        if not path.endswith(".conll"):
            path = f"{os.getcwd()}/../{path}.conll"
        # Create directory if it does not exist
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        print(path)

        CoNLL.write_doc2conll(processed_text, path)

    @staticmethod
    def print(processed_text:str, max_lines=5):
        for sentence in processed_text.sentences[:max_lines]:
            for word in sentence.words:
                print(word.lemma, end=" ")
            print()


def get_labeled_hallucination_by_index(df: pd.DataFrame) -> pd.DataFrame():
    pass


def get_ngrams_around_hallucinations() -> pd.DataFrame():
    pass



if __name__ == "__main__":
    text = "It's the truth, die FPÖ kann scheißen gehn! But if you ask me, the don't even deserve my hatred! 420 blaze it!"
    pipeline_en = Preprocess.get_pipeline("en")
    processed = pipeline_en(text)
    print(processed)
    print(Preprocess.print(processed))
    Preprocess.save_processed_text(processed, "data/output/text_en")
