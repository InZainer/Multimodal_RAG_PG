import json
from pathlib import Path
from typing import List
import logging

class RAGPipeline:
    def __init__(self, config, embed_model, vector_store, colpali_model, logger=None):
        self.config = config
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.colpali_model = colpali_model
        self.logger = logger or logging.getLogger(__name__)

        with open(self.config["paths"]["metadata_index"], "r", encoding='utf-8') as f:
            self.metadata = json.load(f)

        with open(self.config["paths"]["processed_docs"], "r", encoding='utf-8') as f:
            self.all_docs = json.load(f)

        # Определяем максимальное количество токенов для контекста
        self.max_context_tokens = 30000
        self.chunk_size = 3000  # можно регулировать
        self.overlap = 200      # перекрытие между чанками для плавного перехода

    def retrieve_context(self, query: str):
        top_n = self.config["rag_integration"]["retrieval_top_n"]
        q_emb = self.embed_model.encode([query])
        D, I = self.vector_store.search(q_emb, top_n)

        retrieved_docs = []
        for idx in I[0]:
            if idx < 0:
                continue
            meta = self.metadata[idx]
            doc_path = meta["path"]
            for d in self.all_docs:
                if d["path"] == doc_path:
                    retrieved_docs.append(d["text"])
                    break

        # Объединяем все извлеченные тексты в один контекст
        combined_context = "\n\n".join(retrieved_docs)
        return combined_context

    def chunk_text(self, text: str) -> List[str]:
        """
        Разбиваем текст на чанки по self.chunk_size токенов.
        Для упрощения будем считать, что 1 токен ~ 1 слово (на практике надо токенизировать).

        Для более точной оценки нужно использовать self.colpali_model.tokenizer.
        """
        tokens = text.split()
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk = tokens[start:end]
            chunks.append(" ".join(chunk))
            start = end - self.overlap  # перекрытие
            if start < 0:
                start = 0
        return chunks

    def tokenize_length(self, text: str) -> int:
        # Реальный подсчёт токенов через tokenizer
        inputs = self.colpali_model.tokenizer(text, return_tensors='pt')
        return inputs.input_ids.shape[1]

    def split_into_chunks_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        """
        Более точный вариант чанкования по токенам, а не по словам.
        """
        input_ids = self.colpali_model.tokenizer(text, return_tensors='pt').input_ids[0]
        chunks = []
        start = 0
        length = input_ids.shape[0]

        while start < length:
            end = start + max_tokens
            chunk_ids = input_ids[start:end]
            chunk_text = self.colpali_model.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            start = end - overlap_tokens
            if start < 0:
                start = 0
        return chunks

    def answer_query(self, query: str):
        self.logger.info(f"Processing query: {query}")
        context = self.retrieve_context(query)

        # Проверяем длину контекста в токенах
        total_tokens = self.tokenize_length(context)
        if total_tokens > self.max_context_tokens:
            # Разбиваем на чанки
            # Пусть model может обработать 30000 токенов, но для безопасности возьмём меньше, например 10000
            max_tokens_for_chunk = 10000
            overlap_tokens = 300
            chunks = self.split_into_chunks_by_tokens(context, max_tokens_for_chunk, overlap_tokens)

            # Генерируем ответ для каждого чанка и объединяем
            # Логика объединения может быть разной: можно конкатенировать, можно искать итоговый ответ среди частей
            answers = []
            for i, ch in enumerate(chunks):
                self.logger.info(f"Generating answer for chunk {i+1}/{len(chunks)}")
                ans = self.colpali_model.generate_answer(query, ch)
                answers.append(ans)

            # Можно просто вернуть все ответы или попытаться их суммировать
            # Здесь вернем просто конкатенацию (в реальном решении можно использовать RAG повторно)
            final_answer = "\n---\n".join(answers)
        else:
            # Контекст помещается в один запрос
            final_answer = self.colpali_model.generate_answer(query, context)

        return final_answer
