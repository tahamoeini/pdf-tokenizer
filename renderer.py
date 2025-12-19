import os
from typing import List
import fitz
from PIL import Image


def render_page_to_image(pdf_path: str, page_index: int, dpi: int, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = os.path.join(out_dir, f"page_{page_index+1:03}.png")
        pix.save(out_path)
        return out_path
    finally:
        doc.close()


essential_image_ext = ".png"

def extract_embedded_images(pdf_path: str, page_index: int, out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        image_list = page.get_images(full=True)
        for idx, img in enumerate(image_list, start=1):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            # Convert unsupported/CMYK colorspaces to RGB before saving as PNG
            try:
                if pix.n >= 4:
                    rgb = fitz.Pixmap(fitz.csRGB, pix)
                    pix = rgb
            except Exception:
                # Fallback: attempt to create an RGB copy; if this fails, skip image
                try:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                except Exception:
                    continue

            out_path = os.path.join(out_dir, f"page_{page_index+1:03}_img_{idx}{essential_image_ext}")
            pix.save(out_path)
            paths.append(out_path)
    finally:
        doc.close()
    return paths
