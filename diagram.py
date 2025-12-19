import cv2
import numpy as np
from typing import List, Dict, Any
from PIL import Image
import pytesseract
import logging

logger = logging.getLogger(__name__)


def detect_diagrams(image_path: str, ocr_lang: str = 'eng') -> List[Dict[str, Any]]:
    """Detect simple diagram-like structures (boxes and arrows) in an image.

    Returns a list of diagram objects with detected shapes, bounding boxes and OCR'd text.
    This is a heuristic implementation intended as a pragmatic starting point.
    """
    diagrams: List[Dict[str, Any]] = []
    try:
        img = cv2.imread(image_path)
        if img is None:
            return diagrams
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shapes = []
        h, w = gray.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # skip small noise
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Filter out shapes that are too large (likely page border)
            if cw > 0.9 * w and ch > 0.9 * h:
                continue
            # Approximate polygon to guess rectangles
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            shape_type = 'rect' if len(approx) == 4 else 'poly'
            shapes.append({'bbox': (int(x), int(y), int(cw), int(ch)), 'area': int(area), 'type': shape_type})

        # Try to detect arrow-like lines (Hough)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, threshold=50, minLineLength=30, maxLineGap=10)
        edges_list = []
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                edges_list.append({'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)})

        # Map OCR text into shapes
        pil = Image.open(image_path)
        for s in shapes:
            x, y, cw, ch = s['bbox']
            crop = pil.crop((x, y, x + cw, y + ch))
            # Simple OCR
            try:
                text = pytesseract.image_to_string(crop, lang=ocr_lang).strip()
            except Exception as e:
                logger.debug(f"OCR failed on diagram region: {e}")
                text = ''
            s['text'] = text

        if shapes or edges_list:
            diagrams.append({'shapes': shapes, 'edges': edges_list, 'image': image_path})

        return diagrams
    except Exception as e:
        logger.debug(f"Diagram detection failed: {e}")
        return diagrams
