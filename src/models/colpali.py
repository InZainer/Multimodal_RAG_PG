from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

class ColPaliModel:
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Инициализация локальной модели ColPali.
        model_path: путь к модели (либо имя модели на HF Hub).
        device: 'cpu' или 'cuda'
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading ColPali model from {model_path} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def generate_answer(self, query: str, context: str, max_new_tokens=5000, temperature=0.7, top_p=0.9):
        """
        Генерация ответа с учетом контекста.
        Если контекст слишком длинный, он должен быть заранее разделен на чанки.
        Данный метод предполагает, что контекст уже подходит по длине.
        """
        # Формируем финальный промпт
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
        # Ответ содержит изначальный промпт, поэтому можно вычленить только часть после "Ответ:"
        if "Ответ:" in answer:
            answer = answer.split("Ответ:", 1)[-1].strip()
        return answer
