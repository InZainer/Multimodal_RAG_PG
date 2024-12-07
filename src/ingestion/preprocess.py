import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pymorphy2 import MorphAnalyzer
import nltk
from langdetect import detect  # Импортируем langdetect для автоматического определения языка

# Загрузите дополнительные ресурсы NLTK для стоп-слов, если они ещё не загружены
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self, config):
        self.config = config
        self.lemmatizer_en = WordNetLemmatizer()  # Для английского языка
        self.morph = MorphAnalyzer()  # Для русского языка
        # Стоп-слова для английского и русского языков
        self.stopwords_en = set(stopwords.words('english'))
        self.stopwords_ru = set(stopwords.words('russian'))  # Стоп-слова для русского языка

    def preprocess(self, text: str) -> str:
        # Определение языка текста
        language = self.detect_language(text)
        
        rules = self.config.get("normalization_rules", {})

        # Убираем специальные символы, если нужно
        if rules.get("remove_special_characters", True):
            text = re.sub(r'[^\w\s]', '', text)

        # Переводим в нижний регистр
        if rules.get("lowercase", True):
            text = text.lower()

        # Токенизация текста
        tokens = text.split()

        # Убираем стоп-слова
        if language == 'en' and rules.get("remove_stopwords", True):
            tokens = [t for t in tokens if t not in self.stopwords_en]
        elif language == 'ru' and rules.get("remove_stopwords", True):
            tokens = [t for t in tokens if t not in self.stopwords_ru]

        # Лемматизация
        if language == 'en' and rules.get("lemmatization", True):
            tokens = [self.lemmatizer_en.lemmatize(t) for t in tokens]
        elif language == 'ru' and rules.get("lemmatization", True):
            tokens = [self.morph.parse(t)[0].normal_form for t in tokens]

        # Возвращаем результат
        return " ".join(tokens)

    def detect_language(self, text: str) -> str:
        """Определяет язык текста."""
        try:
            lang = detect(text)  # Определяем язык с помощью langdetect
            if lang == 'ru':
                return 'ru'  # Если русский, возвращаем 'ru'
            elif lang == 'en':
                return 'en'  # Если английский, возвращаем 'en'
            else:
                return 'en'  # По умолчанию можно вернуть английский
        except:
            return 'en'  # В случае ошибки, по умолчанию считаем текст английским
