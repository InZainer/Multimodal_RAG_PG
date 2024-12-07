from transformers import PreTrainedTokenizer, LlamaTokenizer
import unicodedata
import re

class GemmaTokenizer(PreTrainedTokenizer):
    """
    Кастомный токенизатор GemmaTokenizer, основанный на LlamaTokenizer.
    Необходимо настроить методы в соответствии с требованиями модели ColPali.
    """

    def __init__(self, vocab_file, merges_file=None, **kwargs):
        """
        Инициализация токенизатора.

        :param vocab_file: Путь к файлу словаря.
        :param merges_file: Путь к файлу объединений (если используется BPE).
        :param kwargs: Дополнительные аргументы.
        """
        super().__init__(**kwargs)
        # Если GemmaTokenizer основан на LlamaTokenizer, можно использовать его как основу
        self.llama_tokenizer = LlamaTokenizer(vocab_file=vocab_file, merges_file=merges_file, **kwargs)

    def _tokenize(self, text):
        """
        Токенизация текста.

        :param text: Входной текст.
        :return: Список токенов.
        """
        # Пример простой токенизации с использованием LlamaTokenizer
        tokens = self.llama_tokenizer.tokenize(text)
        # Здесь можно добавить дополнительные шаги токенизации, специфичные для GemmaTokenizer
        return tokens

    def _convert_token_to_id(self, token):
        """
        Преобразование токена в ID.

        :param token: Токен.
        :return: ID токена.
        """
        return self.llama_tokenizer.convert_tokens_to_ids(token)

    def _convert_id_to_token(self, index):
        """
        Преобразование ID в токен.

        :param index: ID токена.
        :return: Токен.
        """
        return self.llama_tokenizer.convert_ids_to_tokens(index)

    def convert_tokens_to_string(self, tokens):
        """
        Преобразование списка токенов в строку.

        :param tokens: Список токенов.
        :return: Строка.
        """
        return self.llama_tokenizer.convert_tokens_to_string(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Сохранение словаря токенизатора.

        :param save_directory: Директория для сохранения.
        :param filename_prefix: Префикс имени файла.
        :return: Кортеж путей к сохранённым файлам.
        """
        return self.llama_tokenizer.save_vocabulary(save_directory, filename_prefix)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Построение входных данных с особыми токенами.

        :param token_ids_0: Список ID токенов первого предложения.
        :param token_ids_1: Список ID токенов второго предложения (если есть).
        :return: Список ID токенов с особыми токенами.
        """
        return self.llama_tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Получение маски особых токенов.

        :param token_ids_0: Список ID токенов первого предложения.
        :param token_ids_1: Список ID токенов второго предложения (если есть).
        :param already_has_special_tokens: Флаг, указывающий, содержат ли токены уже особые токены.
        :return: Список масок особых токенов.
        """
        return self.llama_tokenizer.get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens)

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Создание ID типов токенов для последовательностей.

        :param token_ids_0: Список ID токенов первого предложения.
        :param token_ids_1: Список ID токенов второго предложения (если есть).
        :return: Список ID типов токенов.
        """
        return self.llama_tokenizer.create_token_type_ids_from_sequences(token_ids_0, token_ids_1)