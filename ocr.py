from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from google.cloud import vision
from google.cloud.vision_v1 import types


def ocr_from_image(image_path):
    """Extract text from an image using Google Cloud Vision API."""

    json_key_path = "secret_key.json"
    # クライアントの初期化
    client = vision.ImageAnnotatorClient.from_service_account_json(json_key_path)

    # 画像の読み込み
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # OCR の実行
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # 最初のテキストアノテーションには全体のテキストが含まれている
    if texts:
        return texts[0].description
    else:
        return ""
