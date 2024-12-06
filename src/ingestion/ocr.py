import pytesseract
from PIL import Image

class OCRProcessor:
    def __init__(self, config):
        self.language = config["ocr_settings"]["language"]
        # Дополнительные настройки можно реализовать
        # DPI можно менять с помощью PIL, если нужно.

    def process_image(self, image_path):
        # OCR для изображений
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=self.language)
        return text
