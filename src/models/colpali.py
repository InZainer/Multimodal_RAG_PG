from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
import torch
import logging

class ColPaliModel:
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Инициализация локальной модели ColPali с явной конфигурацией.
        :param model_path: Путь к локальной модели (директория с config.json, pytorch_model.bin и др.).
        :param device: 'cpu' или 'cuda'.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading ColPali model from {model_path} on {device}")

        # Явно создаём конфигурацию модели LLAMA
        config = LlamaConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_hidden_layers=32,
            vocab_size=32000,
            max_position_embeddings=2048
        )

        # Загружаем модель и токенайзер
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path, config=config)

        # Переносим модель на устройство
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def generate_answer(self, query: str, context: str, max_new_tokens=500, temperature=0.7, top_p=0.9):
        """
        Генерация ответа с учётом контекста.
        :param query: Вопрос пользователя.
        :param context: Контекст из документов.
        :param max_new_tokens: Максимальное количество токенов для генерации.
        :param temperature: Температура для генерации (регулирует креативность модели).
        :param top_p: Top-p sampling.
        :return: Сгенерированный ответ.
        """
        # Формируем промпт
        prompt = f"Вопрос: {query}\nКонтекст:\n{context}\nОтвет:"

        # Токенизация
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        # Генерация текста
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

        # Декодируем ответ
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Извлекаем только часть после "Ответ:"
        if "Ответ:" in answer:
            answer = answer.split("Ответ:", 1)[-1].strip()
        return answer