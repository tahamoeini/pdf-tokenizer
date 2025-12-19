import logging
from typing import Dict, Any
import pytesseract
from PIL import Image
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def ensure_tesseract_available():
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "Tesseract OCR not found. Install from https://github.com/UB-Mannheim/tesseract/wiki "
            "and/or set pytesseract.pytesseract.tesseract_cmd to the tesseract.exe path."
        ) from e


def preprocess_image(img_pil: Image.Image) -> Image.Image:
    try:
        img = np.array(img_pil.convert("L"))
        h, w = img.shape[:2]
        scale = 2 if max(w, h) < 2000 else 1
        if scale != 1:
            img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        img = cv2.fastNlMeansDenoising(img, None, h=10)
        img = cv2.equalizeHist(img)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return Image.fromarray(img)
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}")
        return img_pil


def ocr_image_with_layout(img_pil: Image.Image, lang: str = "eng") -> Dict[str, Any]:
    """Return dict with text and optional layout info."""
    try:
        data = pytesseract.image_to_data(img_pil, lang=lang, output_type=pytesseract.Output.DICT)
        n = len(data.get('level', []))
        lines = {}
        for i in range(n):
            if not data['text'][i].strip():
                continue
            key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
            lines.setdefault(key, []).append(data['text'][i].strip())
        ordered = [" ".join(lines[k]) for k in sorted(lines.keys())]
        raw = pytesseract.image_to_string(img_pil, lang=lang)
        text = "\n".join(ordered) if ordered else raw
        return {"text": text.strip(), "data": data}
    except Exception as e:
        logger.debug(f"OCR layout parse failed: {e}")
        try:
            raw = pytesseract.image_to_string(img_pil, lang=lang)
            return {"text": raw.strip(), "data": None}
        except Exception:
            return {"text": "", "data": None}
