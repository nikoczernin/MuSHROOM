import spacy


class Lemmatizer:
    def __init__(self):
        dummy = 'asdasd'

    @staticmethod
    def lemmatize_text_input(text: str, lang_model: str) -> str:
        nlp = spacy.load(Lemmatizer._get_model_for_lang(lang_model))
        doc = nlp(text)
        return str([token.lemma_ for token in doc])

    @staticmethod
    def _get_model_for_lang(lang: str):
        map_lang_to_spacy_lang_model = {
            'AR': None,
            'ZH': 'zh_core_web_sm',
            'EN': 'en_core_web_sm',
            'DE': 'de_core_news_sm',
            'FI': 'fi_core_news_sm',
            'FR': 'fr_core_news_sm',
            'HI': None,
            'IT': 'it_core_news_sm',
            'ES': 'es_core_news_sm',
            'SV': 'sv_core_news_sm'
        }

        return map_lang_to_spacy_lang_model[lang]
