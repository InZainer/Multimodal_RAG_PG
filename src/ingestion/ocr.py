import concurrent.futures
import easyocr
import logging

class OCRProcessor:
    def __init__(self, config, logger=None):
        self.language = config["ocr_settings"]["language"]
        self.logger = logger or logging.getLogger(__name__)

        # Инициализация EasyOCR с поддержкой GPU
        self.reader = easyocr.Reader(self.language.split('+'), gpu=True)

    def process_image(self, image):
        try:
            self.logger.info("Starting OCR on image")

            # Изменение DPI изображения с помощью PIL (если нужно)
            dpi = 100  # Устанавливаем DPI (можно задать через config)
            image = image.convert('RGB')  # Преобразуем изображение в RGB
            image.info['dpi'] = (dpi, dpi)  # Задаем новый DPI

            # Выполнение OCR с использованием EasyOCR
            result = self.reader.readtext(image)
            text = ' '.join([res[1] for res in result])  # Собираем текст из результатов

            self.logger.info("OCR completed successfully")
            return text
        except Exception as e:
            self.logger.error(f"Error during OCR processing: {e}")
            return ""

    def process_images_parallel(self, image_paths):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self.process_image, image_paths)
        return list(results)