# src/ingestion/preprocess.py
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Не забудьте в реальном решении загрузить ресурсы NLTK
# nltk.download('stopwords')
# nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, config):
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))  # Дополнить русскими стоп-словами при необходимости

    def preprocess(self, text: str) -> str:
        rules = self.config.get("normalization_rules", {})

        if rules.get("remove_special_characters", True):
            text = re.sub(r'[^\w\s]', '', text)

        if rules.get("lowercase", True):
            text = text.lower()

        tokens = text.split()

        if rules.get("remove_stopwords", True):
            tokens = [t for t in tokens if t not in self.stopwords]

        if rules.get("lemmatization", True):
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(tokens)
