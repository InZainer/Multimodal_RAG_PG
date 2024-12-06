import re
import os
from datetime import datetime

class MetadataExtractor:
    def __init__(self, config):
        self.config = config

    def extract_metadata(self, file_path, raw_text):
        meta_config = self.config["metadata_extraction"]["fields"]
        metadata = {}
        # Примеры - в реальном решении могут быть специальные библиотеки для метаданных PDF/Docx
        if meta_config.get("author", False):
            metadata["author"] = "Unknown"  # Можно реализовать логику извлечения
        if meta_config.get("tags", False):
            # Поиск тегов по паттерну
            pattern = self.config["metadata_extraction"]["custom_tags_pattern"]
            # Допустим, что теги встречаются как "#Тег: Finance" в тексте
            tags = re.findall(r'\#Тег:\s*(\w+)', raw_text)
            metadata["tags"] = tags if tags else []
        if meta_config.get("document_type", False):
            # Определяем по расширению
            metadata["document_type"] = os.path.splitext(file_path)[1].replace('.', '')
        if meta_config.get("created_date", False):
            metadata["created_date"] = datetime.now().isoformat()
        if meta_config.get("last_modified_date", False):
            metadata["last_modified_date"] = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

        return metadata
