# src/ingestion/extractors.py
import fitz  # PyMuPDF
import docx
from pathlib import Path
from .pptx_extractor import PPTXExtractor
from .ocr import OCRProcessor
from pdf2image import convert_from_path

class TextExtractor:
    @staticmethod
    def extract_from_pdf(pdf_path: str, use_ocr=False, ocr_processor=None) -> str:
        text = []
        doc = fitz.open(pdf_path)
        if use_ocr and ocr_processor:
            # Если используем OCR, конвертируем страницы в изображения и применяем OCR
            logger = ocr_processor.logger
            logger.info(f"Using OCR to extract text from PDF: {pdf_path}")
            images = convert_from_path(pdf_path, dpi=300)
            for page_num, img in enumerate(images, start=1):
                logger.info(f"Processing OCR for page {page_num}")
                page_text = ocr_processor.process_image(img)
                text.append(page_text)
        else:
            # Извлечение текста напрямую из PDF
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
    def extract_from_pptx(pptx_path: str) -> str:
        return PPTXExtractor.extract_text(pptx_path)

    @staticmethod
    def extract_from_txt(txt_path: str) -> str:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def extract_raw(file_path: Path, use_ocr=False, ocr_processor=None) -> str:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return TextExtractor.extract_from_pdf(str(file_path), use_ocr, ocr_processor)
        elif ext == ".docx":
            return TextExtractor.extract_from_docx(str(file_path))
        elif ext == ".pptx":
            return TextExtractor.extract_from_pptx(str(file_path))
        elif ext == ".txt":
            return TextExtractor.extract_from_txt(str(file_path))
        else:
            return ""
