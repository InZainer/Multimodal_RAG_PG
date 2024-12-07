# src/main.py
import sys
import json
from pathlib import Path
from common.logger import get_logger
from common.utils import load_config, save_json
from ingestion.ocr import OCRProcessor
from ingestion.extractors import TextExtractor
from ingestion.preprocess import TextPreprocessor
from ingestion.metadata import MetadataExtractor
from ingestion.formula_extractor import FormulaExtractor
from ingestion.table_extractor import TableExtractor
from indexing.embeddings import EmbeddingModel
from indexing.vector_store import VectorStore
from models.colpali import ColPaliModel
from rag.pipeline import RAGPipeline
import os

def process_documents(config, logger):
    data_dir = Path(config["paths"]["data_dir"])
    ocr = OCRProcessor(config)
    preprocessor = TextPreprocessor(config)
    metadata_extractor = MetadataExtractor(config)
    formula_extractor = FormulaExtractor(config)
    table_extractor = TableExtractor(config)

    processed_docs = []

    for file in data_dir.iterdir():
        if file.is_file():
            ext = file.suffix.lower()
            logger.info(f"Processing file: {file}")
            if ext in [".png", ".jpg", ".jpeg", ".tiff"]:
                raw_text = ocr.process_image(str(file))
            else:
                raw_text = TextExtractor.extract_raw(file)

            tables = []
            if ext == ".pdf":
                tables = table_extractor.extract_tables_from_pdf(str(file))

            # Формулы
            formulas = formula_extractor.extract_formulas(raw_text)
            if formulas:
                raw_text += "\n\n" + "\n".join(["[FORMULA]: " + f for f in formulas])

            norm_text = preprocessor.preprocess(raw_text)
            doc_metadata = metadata_extractor.extract_metadata(str(file), raw_text)

            if tables:
                doc_metadata["tables"] = tables

            processed_docs.append({
                "path": str(file),
                "text": norm_text,
                "metadata": doc_metadata
            })

    save_json(processed_docs, config["paths"]["processed_docs"])
    return processed_docs

def build_index(config, logger):
    with open(config["paths"]["processed_docs"], "r", encoding="utf-8") as f:
        docs = json.load(f)

    embed_model = EmbeddingModel()
    texts = [d["text"] for d in docs]
    embeddings = embed_model.encode(texts)

    dimension = embeddings.shape[1]
    vector_store = VectorStore(dimension)
    vector_store.add(embeddings)
    vector_store.save(config["paths"]["vector_index"])

    metadata = [{"path": d["path"], "metadata": d["metadata"]} for d in docs]
    save_json(metadata, config["paths"]["metadata_index"])

    return embed_model, vector_store

def main():
    config = load_config("config/config.json")
    logger = get_logger(name="RAGSystem")

    # Предобработка и индексация
    process_documents(config, logger)
    embed_model, vector_store = build_index(config, logger)

    # Загрузка модели ColPali (Qwen) с явной конфигурацией
    model_path = "qwen2-vl-7b-instruct"  # Путь к директории с моделью

    # Инициализация ColPaliModel
    colpali_model = ColPaliModel(
        model_path=model_path,
        device="cuda"  # Используйте "cpu", если GPU недоступен
    )

    # Загрузка индекса
    vector_store_loaded = VectorStore(768)
    vector_store_loaded.load(config["paths"]["vector_index"])

    # Инициализация RAG пайплайна
    rag_pipeline = RAGPipeline(config, embed_model, vector_store_loaded, colpali_model, logger)

    # Пример запроса
    user_query = "Долевое участие металлов в EBITDA компании? Какой процент принес Ni и Cu? Учесть контекст документов."
    answer = rag_pipeline.answer_query(user_query)
    print("Ответ модели:")
    print(answer)

if __name__ == "__main__":
    main()
