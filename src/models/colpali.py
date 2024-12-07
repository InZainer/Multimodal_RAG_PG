from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration
import torch
import logging

class QwenModel:
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Инициализация модели Qwen с использованием модели Qwen/Qwen2-VL-7B-Instruct.
        :param model_path: Путь к директории модели.
        :param device: Устройство ('cuda' или 'cpu').
        """
        self.logger = logging.getLogger(__name__)
        try:
            self.logger.info(f"Loading Qwen model from {model_path} on {device}")

            # Загрузка конфигурации модели
            config = AutoConfig.from_pretrained(model_path)

            # Загрузка токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

            # Загрузка модели с использованием 16-битной точности для экономии памяти
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            # Перенос модели на устройство
            self.device = device
            self.model.to(self.device)
            self.model.eval()

            self.logger.info("Qwen model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading Qwen model: {e}")
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