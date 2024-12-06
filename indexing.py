import json
import numpy as np
from sentence_transformers import SentenceTransformer
# Предполагаем наличие векторного хранилища (Chroma, Faiss и т.д.)
# Здесь для упрощения просто сохраним в файлы.
import faiss


def load_processed_docs(path="processed_docs.json"):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    docs = load_processed_docs()
    # Пример использования модели для эмбеддингов (надо установить sentence-transformers)
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    texts = [doc["text"] for doc in docs]
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Создаём индекс FAISS для векторного поиска
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, "indexes/vector.index")

    # Сохраним метаданные отдельно
    metadata = [{"path": d["path"], "metadata": d["metadata"]} for d in docs]
    with open("indexes/metadata.json", "w", encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
