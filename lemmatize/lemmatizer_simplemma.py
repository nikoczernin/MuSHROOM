import simplemma


# simplemma misses the following languages
# Arabic (Modern standard)
# Chinese (Mandarin)


class Lemmatizer:

    @staticmethod
    def lemmatize_text_input(text: str, lang="en") -> str:
        """
        Lemmatizes the input text using a specified language model.

        Parameters:
        - text (str): The text to lemmatize.
        - languages (tuple/str): The languages included in the text.

        Returns:
        - str: A list of lemmatized words in string format.
        """
        # if languages has been passed as a list, turn it into a tuple
        if isinstance(lang, list):
            languages = tuple(lang)
        # languages should all be lowercase
        if isinstance(lang, tuple):
            lang = tuple(lang.lower() for lang in lang)
        else:
            lang = lang.lower()
        # return the lemmatized tokens as a string
        return str(simplemma.text_lemmatizer(text, lang=lang))


if __name__ == "__main__":
    text = "Yall hate to hear it but he said auf Wiedersehen mein Süßer!"
    print(Lemmatizer.lemmatize_text_input(text, lang=["DE", "en"]))
