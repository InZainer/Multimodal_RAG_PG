# src/ingestion/formula_extractor.py
import re

class FormulaExtractor:
    def __init__(self, config):
        self.config = config

    def extract_formulas(self, text: str):
        # Пример: ищем формулы в стиле $...$ или $$...$$
        # Для реального решения: использовать ML-модели или более сложные алгоритмы.
        if not self.config["formula_extraction"]["enabled"]:
            return []
        pattern = r'(\${1,2}.*?\${1,2})'
        formulas = re.findall(pattern, text, flags=re.DOTALL)
        return formulas
