from transformers import LlamaConfig, LlamaForCausalLM
import torch
import logging
from models.gemma_tokenizer import GemmaTokenizer  # Убедитесь, что путь корректен

class ColPaliModel:
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Инициализация модели ColPali с использованием GemmaTokenizer.
        :param model_path: Путь к директории модели.
        :param device: Устройство ('cuda' или 'cpu').
        """
        self.logger = logging.getLogger(__name__)
        try:
            self.logger.info(f"Loading ColPali model from {model_path} on {device}")

            # Загрузка конфигурации модели
            config = LlamaConfig.from_pretrained(model_path)

            # Загрузка кастомного токенизатора
            self.tokenizer = GemmaTokenizer.from_pretrained(model_path)

            # Загрузка модели с использованием 16-битной точности для экономии памяти
            self.model = LlamaForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            # Переносим модель на устройство
            self.device = device
            self.model.to(self.device)
            self.model.eval()

            self.logger.info("Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading ColPali model: {e}")
            raise e

    def generate_answer(self, query: str, context: str, max_new_tokens: int = 500, temperature: float = 0.7, top_p: float = 0.9):
        """
        Генерация ответа на основе запроса и контекста.
        :param query: Вопрос пользователя.
        :param context: Контекст из документов.
        :param max_new_tokens: Максимальное количество генерируемых токенов.
        :param temperature: Температура генерации.
        :param top_p: Top-p sampling.
        :return: Сгенерированный ответ.
        """
        try:
            prompt = f"Вопрос: {query}\nКонтекст:\n{context}\nОтвет:"
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Ответ:" in answer:
                answer = answer.split("Ответ:", 1)[-1].strip()
            return answer
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return "Извините, произошла ошибка при генерации ответа."