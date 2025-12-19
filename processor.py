import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

import pdfplumber
from PIL import Image

from config import Config
from renderer import render_page_to_image, extract_embedded_images
from ocr import ensure_tesseract_available, preprocess_image, ocr_image_with_layout
from exporters import TextExporter, JsonExporter, MarkdownExporter

logger = logging.getLogger(__name__)


@dataclass
class PageResult:
    page_number: int
    text: str
    text_source: str  # "embedded" or "ocr"
    image_files: List[str] = field(default_factory=list)
    ocr_blocks: Optional[Dict[str, Any]] = None


@dataclass
class DocumentResult:
    filename: str
    input_path: str
    page_count: int
    metadata: Dict[str, Any]
    pages: List[PageResult]
    produced_files: List[str]


class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config

    def process(self, pdf_path: str) -> DocumentResult:
        filename = os.path.basename(pdf_path)
        doc_stem = os.path.splitext(filename)[0]
        base_dir = os.path.join(self.config.output_dir, doc_stem)
        images_dir = os.path.join(base_dir, "images")
        extracted_dir = os.path.join(base_dir, "extracted_images")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(extracted_dir, exist_ok=True)

        # Tesseract check will be deferred until OCR is actually required

        # Metadata
        page_count = 0
        metadata: Dict[str, Any] = {}
        try:
            import fitz
            d = fitz.open(pdf_path)
            metadata = d.metadata or {}
            page_count = len(d)
            d.close()
        except Exception as e:
            logger.warning(f"Failed to read metadata: {e}")

        pages: List[PageResult] = []
        produced: List[str] = []

        with pdfplumber.open(pdf_path) as pdf:
            page_count = page_count or len(pdf.pages)
            for idx, page in enumerate(pdf.pages):
                page_num = idx + 1
                embedded = page.extract_text() or ""
                embedded_norm = "\n".join([ln.strip() for ln in embedded.splitlines()]).strip()
                use_ocr = self.config.ocr_enabled and (len(embedded_norm) < self.config.embedded_text_threshold)

                page_render_path = render_page_to_image(pdf_path, idx, self.config.dpi, images_dir)
                produced.append(page_render_path)

                image_files = []
                # Always include the rendered page image first
                image_files.append(os.path.relpath(page_render_path, base_dir).replace('\\', '/'))

                # Extract embedded images (optional but useful)
                if self.config.preserve_images:
                    embeds = extract_embedded_images(pdf_path, idx, extracted_dir)
                    for p in embeds:
                        produced.append(p)
                    image_files.extend([os.path.relpath(p, base_dir).replace('\\', '/') for p in embeds])

                text_source = "embedded"
                text_value = embedded_norm
                ocr_blocks = None

                if use_ocr:
                    ensure_tesseract_available()
                    img = Image.open(page_render_path)
                    pre = preprocess_image(img)
                    ocr = ocr_image_with_layout(pre, lang=self.config.lang)
                    text_value = (ocr.get("text") or "").strip()
                    text_source = "ocr"
                    ocr_blocks = ocr.get("data")

                pages.append(PageResult(
                    page_number=page_num,
                    text=text_value,
                    text_source=text_source,
                    image_files=image_files,
                    ocr_blocks=ocr_blocks,
                ))

        # Exporters
        produced_files: List[str] = []
        textexp = TextExporter(base_dir)
        full_text_acc: List[str] = []
        for p in pages:
            full_text_acc.append(p.text)
            produced_files.append(textexp.write_page(p.page_number, p.text))
        produced_files.append(textexp.write_full("\n\n".join(full_text_acc)))

        # JSON
        if self.config.wants("json"):
            jexp = JsonExporter(base_dir)
            payload = {
                "metadata": {
                    "input_filename": filename,
                    "page_count": page_count,
                    "created_at": datetime.now().isoformat(),
                    "tool_versions": self._tool_versions(),
                },
                "pages": [
                    {
                        "page_number": p.page_number,
                        "text_source": p.text_source,
                        "text": p.text,
                        "image_files": p.image_files,
                    }
                    for p in pages
                ],
            }
            produced_files.append(jexp.write(payload))

        # Markdown
        if self.config.wants("md"):
            mexp = MarkdownExporter(base_dir)
            produced_files.append(mexp.write([
                {
                    "page_number": p.page_number,
                    "text": p.text,
                    "image_files": p.image_files,
                } for p in pages
            ], metadata={"title": metadata.get("title") or filename, "page_count": page_count, "filename": filename}))

        produced_files.extend(produced)

        return DocumentResult(
            filename=filename,
            input_path=pdf_path,
            page_count=page_count,
            metadata=metadata,
            pages=pages,
            produced_files=produced_files,
        )

    def _tool_versions(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            import fitz
            out["pymupdf"] = getattr(fitz, "__doc__", "")
        except Exception:
            pass
        try:
            import pdfplumber
            out["pdfplumber"] = getattr(pdfplumber, "__version__", "")
        except Exception:
            pass
        try:
            import pytesseract
            out["pytesseract"] = str(pytesseract.get_tesseract_version())
        except Exception:
            out["pytesseract"] = "not available"
        return out
