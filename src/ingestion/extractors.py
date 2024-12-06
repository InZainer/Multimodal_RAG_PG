import fitz  # PyMuPDF for PDF
import docx
from pathlib import Path

class TextExtractor:
    @staticmethod
    def extract_from_pdf(pdf_path: str) -> str:
        text = []
        doc = fitz.open(pdf_path)
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text)

    @staticmethod
    def extract_from_docx(docx_path: str) -> str:
        doc = docx.Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)

    @staticmethod
    def extract_from_txt(txt_path: str) -> str:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def extract_raw(file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return TextExtractor.extract_from_pdf(str(file_path))
        elif ext == ".docx":
            return TextExtractor.extract_from_docx(str(file_path))
        elif ext in [".txt"]:
            return TextExtractor.extract_from_txt(str(file_path))
        else:
            # Неизвестный формат
            return ""
