import os
import json
import subprocess
from pathlib import Path
import pytesseract
from PIL import Image
import fitz  # PyMuPDF for PDF
import docx
import re


# Предполагаем что есть какие-то функции для лемматизации, удаления стоп-слов и т.д.
# Заглушки для примера.
def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_text_from_pdf(pdf_path):
    text = []
    doc = fitz.open(pdf_path)
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)


def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)


def ocr_image(image_path, lang="eng"):
    # OCR для изображений
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    return text


def normalize_text(text, config):
    # Пример нормализации по правилам из config:
    rules = config.get("normalization_rules", {})
    if rules.get("lowercase", False):
        text = text.lower()
    if rules.get("remove_special_characters", False):
        text = re.sub(r'[^\w\s]', '', text)
    # Добавьте лемматизацию и удаление стоп-слов по необходимости
    return text


def extract_metadata(file_path, config):
    # Пример: читаем метаданные из имени файла, или у pdf извлекаем дату создания.
    # Здесь все упрощено.
    metadata = {}
    if config["metadata_extraction"]["fields"].get("author"):
        metadata["author"] = "Unknown"
    if config["metadata_extraction"]["fields"].get("tags"):
        metadata["tags"] = ["example_tag"]
    return metadata


def process_file(file_path, config):
    ext = file_path.suffix.lower()
    raw_text = ""
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(str(file_path))
    elif ext == ".docx":
        raw_text = extract_text_from_docx(str(file_path))
    elif ext in [".png", ".jpg", ".jpeg"]:
        # OCR для изображений
        lang = config["ocr_settings"]["language"]
        raw_text = ocr_image(str(file_path), lang=lang)
    else:
        # Попытка прочитать как текстовый файл
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except:
            raw_text = ""

    norm_text = normalize_text(raw_text, config)
    metadata = extract_metadata(file_path, config)
    return {
        "path": str(file_path),
        "text": norm_text,
        "metadata": metadata
    }


if __name__ == "__main__":
    config = load_config("config.json")
    data_dir = Path("data")
    processed_documents = []
    for file in data_dir.iterdir():
        if file.is_file():
            doc_data = process_file(file, config)
            processed_documents.append(doc_data)

    # Сохраняем промежуточный результат для следующего шага (индексации)
    with open("processed_docs.json", "w", encoding='utf-8') as f:
        json.dump(processed_documents, f, ensure_ascii=False, indent=2)
