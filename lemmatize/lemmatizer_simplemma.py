from


class Lemmatizer:
    def __init__(self):
        # Initialize the Lemmatizer class. The `dummy` variable here is currently unused.
        # If you need to store instance-specific data in the future, it can be removed or replaced.
        dummy = 'asdasd'

    @staticmethod
    def lemmatize_text_input(text: str) -> str:
        """
        Lemmatizes the input text using a specified language model.

        Parameters:
        - text (str): The text to lemmatize.

        Returns:
        - str: A list of lemmatized words in string format.
        """

        return ""

    @staticmethod
    def _get_model_for_lang(lang: str):
        """
        Maps a language code to the corresponding spaCy language model.

        Parameters:
        - lang (str): The language code (e.g., 'EN' for English).

        Returns:
        - str or None: The name of the spaCy model if available; None if the language is unsupported.
        """
        # Dictionary mapping of language codes to spaCy language models.
        # Models set to None indicate unsupported or unavailable models in this mapping.
        map_lang_to_spacy_lang_model = {
            'AR': None,  # Arabic (no model specified)
            'ZH': 'zh_core_web_sm',  # Chinese
            'EN': 'en_core_web_sm',  # English
            'DE': 'de_core_news_sm',  # German
            'FI': 'fi_core_news_sm',  # Finnish
            'FR': 'fr_core_news_sm',  # French
            'HI': None,  # Hindi (no model specified)
            'IT': 'it_core_news_sm',  # Italian
            'ES': 'es_core_news_sm',  # Spanish
            'SV': 'sv_core_news_sm'  # Swedish
        }

        # Return the model name for the given language code.
        return map_lang_to_spacy_lang_model[lang]
