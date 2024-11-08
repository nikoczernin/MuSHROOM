import pandas as pd
import stanza
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from stanza.pipeline.multilingual import MultilingualPipeline
import os

# Download required Stanza language models
stanza.download(lang="multilingual")  # Download model for multilingual processing


class Preprocess:
    # Class attribute to track downloaded languages


    def __init__(self):
        # keep a dict of all pipelines
        self.PIPELINES = {
            # Initialize with a MultilingualPipeline for processing multiple languages
            "auto": MultilingualPipeline(processors='tokenize,lemma,pos')
        }


    def update_languages(self, langs):
        """
        Adds a specified language model if not already downloaded,
        and updates the LANGUAGES list.

        Parameters:
        - lang (str): Language code (e.g., 'en' for English)
        """
        for lang in list(set(langs)):
            lang = lang.lower() # stanza languages are all lowercase
            if lang not in self.PIPELINES.keys():
                print(f"Downloading missing language: {lang}")
                stanza.download(lang)
                self.PIPELINES[lang] = stanza.Pipeline(lang, processors='tokenize,lemma,pos')




    def preprocess(self, docs: list, lang="auto"):
        """
        Processes a list of text documents, converting each to a Stanza Document format.

        Parameters:
        - docs (list): List of strings (documents) to preprocess.
        - lang (str): Language code (default is "en" for English).

        Returns:
        - List[Document]: List of processed Document objects.
        """
        # Convert each input text to a Stanza Document object
        docs = [Document([], text=text) for text in docs]
        # Return the processed documents using the initialized pipeline
        # if no language parameter was given, use the autodetect multilingual pipeline
        # otherwise use the wanted language
        # print(lang)
        # print(type(lang))
        if isinstance(lang, str):
            processed_text = self.PIPELINES[lang.lower()](docs)
        else:
            processed_text = self.PIPELINES["auto"](docs)

        return processed_text


    @staticmethod
    def save_processed_text(processed_text, path: str):
        """
        Saves the processed text in CoNLL format to the specified path.

        Parameters:
        - processed_text: The processed text (list of Stanza Document objects).
        - path (str): Path to save the .conll file.
        """

        # Create the directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)


        # Save the processed text in CoNLL format
        print("Saving to", path)
        for i,doc in enumerate(processed_text):
            CoNLL.write_doc2conll(doc, f"{path}/{i}.conllu")
        print("\tSaving successful!")

    @staticmethod
    def print(processed_text, max_lines=None):
        """
        Prints the lemmatized words in the processed text up to max_lines.

        Parameters:
        - processed_text: The processed text (list of Stanza Document objects).
        - max_lines (int or None): Maximum number of lines to print. Prints all lines if None.
        """

        # Print lemmatized words, observing the line limit
        for doc in processed_text:
            for sentence in doc.sentences:
                for word in sentence.words:
                    print(word.lemma, end=" ")
                print()


# Placeholder functions for specific data processing tasks
def get_labeled_hallucination_by_index(df: pd.DataFrame) -> pd.DataFrame:
    pass


def get_ngrams_around_hallucinations() -> pd.DataFrame:
    pass


def test():
    # Sample text input for preprocessing
    text = [
        "Es ist die blanke Wahrheit: die FPÖ kann scheißen gehn!",
        # "But if you ask me, the don't even deserve my hatred!",
        "جليقة، التي تعرف الآن باسم كوريا الجنوبية، تتأ",
        "सजायाफ्ता कैदियों को टेलीविज़न(टेलीविज़न) सीरीज़ 'जेल में बंद' के लिए वाइट कॉलर टीवी के रूप में पेश किया जाता है। वाइट कॉलर टेलीविज़न शृंखला में मुख़्य किरदार में पुलिस(पुलिस जे) द्वारा बन्दी बनाये जाने पर जेल से बचने के लिए उन्हें शारीरिक व मानसिक रूप से अक्षम दिखाया गया है। 'जेल में बंद' कहानी कहने का एक अनूठा माध्यम है, क्योंकि यह एक जेल में कैदी की स्थिति को दर्शाता है। जेल में कैदी के रूप में एक कैदी के जीवन के आसपास केंद्रित है। कैदी को जेल में किसी अपराध के लिए दोषी ठहराया गया है और वह अब अपनी सजा का काट रहा है। वाइट कॉलर टीवी सीरीज़ में एक जेल में कैदी की स्थिति को दर्शाया गया है, जिसमें एक कैदी का जीवन दिन - प्रति - दिन चलता रहता है और उसकी मानसिक, शारीरिक और भावनात्मक समस्याएं कैदी पर उनके अपने नियंत्रण से परे रहती हैं। कुल मिलाकर, वाइट कॉलर टीवी सीरीज़ में मुख्य पात्रों में से एक को जेल कैद के रूप में दर्शाया गया है, जो वास्तविकता के अनुरूप है। कैदी को जेल में कैद कर दिया गया है, जिसमें उसे जेल अधिकारियों, डॉक्टरों और समाज द्वारा निभाई गई एक भ्रष्ट और हिंसक भूमिका के लिए दोषी ठहराया गया है। यह जेल में कैद कहानी के सार का प्रतिबिंब है, और उस जेल जीवन का समाज और दुनिया के लिए एक संदेश प्रदान करता है",
        "420 blaze it!"
    ]

    # Initialize the Preprocess object and process the text
    preprocessor = Preprocess()
    processed = preprocessor.preprocess(text)

    # Print processed output
    print(type(processed))
    Preprocess.print(processed)

    # Save processed output in CoNLL format
    preprocessor.save_processed_text(processed, "../data/output/test_en")

    # read in the saved data to make sure it works
    read_path = "../data/output/test_en"
    docs = [CoNLL.conll2doc(f"{read_path}/{path}") for path in os.listdir(read_path)]
    print(docs)


if __name__ == "__main__":
    test()