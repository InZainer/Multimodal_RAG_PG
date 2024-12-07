from models.colpali import QwenModel

def test_model():
    model_path = "qwen2-vl-7b-instruct"  # Обновите путь к вашей модели
    device = "cuda"  # Или "cpu" если нет GPU

    try:
        model = QwenModel(model_path=model_path, device=device)
        print("Model and tokenizer loaded successfully.")

        # Тестовый запрос
        query = "Долевое участие металлов в EBITDA компании? Какой процент принес Ni и Cu?"
        context = "Ваш тестовый контекст из документов."

        answer = model.generate_answer(query, context)
        print("Ответ модели:")
        print(answer)
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    test_model()