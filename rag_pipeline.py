import json
import faiss
import numpy as np
from colpali_model import ColPaliModel
from sentence_transformers import SentenceTransformer

def load_config(path="config.json"):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

def load_metadata(path="indexes/metadata.json"):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

def search_top_n(query, model, index, metadata, n=5):
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, n)
    results = []
    for idx in I[0]:
        if idx < 0:
            continue
        meta = metadata[idx]
        results.append(meta)
    return results

if __name__ == "__main__":
    config = load_config()
    model_version = config["rag_integration"]["colpali_model_version"]
    top_n = config["rag_integration"]["retrieval_top_n"]

    # Загрузка ColPali модели
    colpali = ColPaliModel(model_version)

    # Загрузка векторного индекса и метаданных
    index = faiss.read_index("indexes/vector.index")
    metadata = load_metadata()

    # Загрузка модели эмбеддингов
    embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Пример запроса
    user_query = "Каковы стандарты оформления документов?"

    # Поиск релевантных документов
    results = search_top_n(user_query, embed_model, index, metadata, n=top_n)

    # Собираем контекст из найденных документов
    # В реальном решении нужно не только метаданные, но и тексты документов.
    # Предположим, что тексты уже в памяти или их можно подгрузить.
    # Для простоты возьмём текст из processed_docs.json.
    with open("processed_docs.json", "r", encoding='utf-8') as f:
        all_docs = json.load(f)
    context_docs = []
    for r in results:
        # Ищем соответствующий doc
        for d in all_docs:
            if d["path"] == r["path"]:
                context_docs.append(d["text"])
                break

    # Соединяем тексты для контекста
    combined_context = "\n\n".join(context_docs)

    # Генерация ответа с помощью ColPali
    answer = colpali.generate_answer(user_query, combined_context)
    print(answer)
