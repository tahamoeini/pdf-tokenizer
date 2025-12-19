import os
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

import pdfplumber
from PIL import Image

from config import Config
from renderer import render_page_to_image, extract_embedded_images
from ocr import ensure_tesseract_available, preprocess_image, ocr_image_with_layout
from exporters import TextExporter, JsonExporter, MarkdownExporter
from diagram import detect_diagrams

logger = logging.getLogger(__name__)


@dataclass
class PageResult:
    page_number: int
    text: str
    text_source: str  # "embedded" or "ocr"
    image_files: List[str] = field(default_factory=list)
    ocr_blocks: Optional[Dict[str, Any]] = None
    embedded_images: List[Dict[str, Any]] = field(default_factory=list)
    diagrams: List[Dict[str, Any]] = field(default_factory=list)
    # Source tracking
    embedded_text: str = ""
    ocr_text: str = ""
    extraction_time_ms: int = 0
    source_details: Dict[str, Any] = field(default_factory=dict)


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
                page_start_time = time.time()
                
                # Extract embedded text first
                embedded = page.extract_text() or ""
                embedded_norm = "\n".join([ln.strip() for ln in embedded.splitlines()]).strip()
                
                # Decide if OCR is needed
                use_ocr = self.config.ocr_enabled and (len(embedded_norm) < self.config.embedded_text_threshold)

                # OPTIMIZATION: Only render page if OCR fallback is needed
                page_render_path = None
                if use_ocr or self.config.preserve_images:
                    page_render_path = render_page_to_image(pdf_path, idx, self.config.dpi, images_dir)
                    produced.append(page_render_path)

                image_files = []
                # Include the rendered page image if it was produced
                if page_render_path:
                    image_files.append(os.path.relpath(page_render_path, base_dir).replace('\\', '/'))

                # Extract embedded images (optional but useful)
                embedded_images = []
                embeds = []
                if self.config.preserve_images and page_render_path:
                    embeds = extract_embedded_images(pdf_path, idx, extracted_dir)
                    for p in embeds:
                        produced.append(p)
                    rels = [os.path.relpath(p, base_dir).replace('\\', '/') for p in embeds]
                    image_files.extend(rels)

                    # OCR each embedded image and collect extracted text
                    embedded_images = []
                    if self.config.ocr_enabled:
                        for p, rel in zip(embeds, rels):
                            try:
                                ensure_tesseract_available()
                                img_e = Image.open(p)
                                pre_e = preprocess_image(img_e)
                                ocr_e = ocr_image_with_layout(pre_e, lang=self.config.lang)
                                text_e = (ocr_e.get('text') or '').strip() if isinstance(ocr_e, dict) else (ocr_e or '')
                                embedded_images.append({
                                    'path': rel,
                                    'text': text_e,
                                    'ocr_data': ocr_e.get('data') if isinstance(ocr_e, dict) else None,
                                })
                            except Exception as e:
                                logger.debug(f"Embedded image OCR failed for {p}: {e}")
                                embedded_images.append({'path': rel, 'text': '', 'ocr_data': None})
                    else:
                        # If OCR disabled, include path placeholders
                        embedded_images = [{'path': rel, 'text': '', 'ocr_data': None} for rel in rels]

                # Run diagram detection on the rendered page image and embedded images
                diagrams_found = []
                if page_render_path:
                    try:
                        # page-level diagrams
                        page_diags = detect_diagrams(page_render_path, ocr_lang=self.config.lang)
                        if page_diags:
                            diagrams_found.extend(page_diags)
                        # embedded image diagrams
                        if self.config.preserve_images:
                            for p in embeds:
                                try:
                                    ed = detect_diagrams(p, ocr_lang=self.config.lang)
                                    if ed:
                                        diagrams_found.extend(ed)
                                except Exception:
                                    continue
                    except Exception as e:
                        logger.debug(f"Diagram detection error: {e}")

                # Separate text sources
                embedded_text = embedded_norm
                ocr_text = ""
                text_source = "embedded"
                text_value = embedded_norm
                ocr_blocks = None

                if use_ocr:
                    ensure_tesseract_available()
                    if not page_render_path:
                        page_render_path = render_page_to_image(pdf_path, idx, self.config.dpi, images_dir)
                        produced.append(page_render_path)
                        image_files.insert(0, os.path.relpath(page_render_path, base_dir).replace('\\', '/'))
                    img = Image.open(page_render_path)
                    pre = preprocess_image(img)
                    ocr = ocr_image_with_layout(pre, lang=self.config.lang)
                    ocr_text = (ocr.get("text") or "").strip()
                    text_source = "ocr"
                    text_value = ocr_text
                    ocr_blocks = ocr.get("data")

                # Track extraction timing and source details
                extraction_time = int((time.time() - page_start_time) * 1000)  # milliseconds
                source_details = {
                    "embedded_text_length": len(embedded_text),
                    "ocr_used": use_ocr,
                    "threshold": self.config.embedded_text_threshold,
                    "page_rendered": page_render_path is not None,
                    "extraction_time_ms": extraction_time,
                    "text_source": text_source,
                }

                pages.append(PageResult(
                    page_number=page_num,
                    text=text_value,
                    text_source=text_source,
                    image_files=image_files,
                    ocr_blocks=ocr_blocks,
                    embedded_images=embedded_images,
                    diagrams=diagrams_found,
                    embedded_text=embedded_text,
                    ocr_text=ocr_text,
                    extraction_time_ms=extraction_time,
                    source_details=source_details,
                ))

        # Exporters
        produced_files: List[str] = []
        textexp = TextExporter(base_dir)
        full_text_acc: List[str] = []
        for p in pages:
            # Append page text and any embedded image OCR text
            page_parts: List[str] = []
            if p.text:
                page_parts.append(p.text)
            for ei in getattr(p, 'embedded_images', []):
                if ei.get('text'):
                    page_parts.append(f"[Embedded image: {ei.get('path')}]")
                    page_parts.append(ei.get('text'))

            page_text_combined = "\n\n".join(page_parts).strip()
            full_text_acc.append(page_text_combined)
            produced_files.append(textexp.write_page(p.page_number, page_text_combined))
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
                        "embedded_text": getattr(p, 'embedded_text', ''),
                        "ocr_text": getattr(p, 'ocr_text', ''),
                        "source_details": getattr(p, 'source_details', {}),
                        "extraction_time_ms": getattr(p, 'extraction_time_ms', 0),
                        "image_files": p.image_files,
                        "embedded_images": p.embedded_images,
                        "diagrams": p.diagrams,
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
                    "embedded_text": getattr(p, 'embedded_text', ''),
                    "ocr_text": getattr(p, 'ocr_text', ''),
                    "source_details": getattr(p, 'source_details', {}),
                    "image_files": p.image_files,
                    "embedded_images": p.embedded_images,
                    "diagrams": p.diagrams,
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
