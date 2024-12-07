# src/ingestion/ocr.py
import pytesseract
from PIL import Image
import logging


class OCRProcessor:
    def __init__(self, config, logger=None):
        self.language = config["ocr_settings"]["language"]
        self.logger = logger or logging.getLogger(__name__)

    def process_image(self, image):
        try:
            self.logger.info("Starting OCR on image")

            # Дополнительные настройки
            # Изменение DPI изображения с помощью PIL (если нужно)
            dpi = 300  # Устанавливаем DPI в 300 (можно настроить через config)
            image = image.convert('RGB')  # Преобразуем изображение в RGB
            image.info['dpi'] = (dpi, dpi)  # Задаем новый DPI

            # Выполняем OCR на изображении
            text = pytesseract.image_to_string(image, lang=self.language)

            self.logger.info("OCR completed successfully")
            return text
        except Exception as e:
            self.logger.error(f"Error during OCR processing: {e}")
            return ""
