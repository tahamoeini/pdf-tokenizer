import os
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

import pdfplumber
from PIL import Image
import concurrent.futures
import multiprocessing
try:
    import psutil
except Exception:
    psutil = None

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
    runtime_stats: Dict[str, Any] = field(default_factory=dict)


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
        doc_start_time = time.time()
        resource_snapshot = {'cpu_percent': None, 'available_ram_mb': None}

        with pdfplumber.open(pdf_path) as pdf:
            page_count = page_count or len(pdf.pages)
            # Two-phase approach: quick embedded-text scan, then parallel heavy processing
            def analyze_text_quality(text_value: str) -> Dict[str, Any]:
                normalized = text_value or ""
                length = len(normalized)
                alnum = sum(1 for ch in normalized if ch.isalnum())
                ratio = (alnum / length) if length else 0.0
                min_required = max(
                    getattr(self.config, 'embedded_text_threshold', 0),
                    getattr(self.config, 'ocr_text_min_length', 0),
                )
                needs_ocr = (length == 0) or (length < min_required) or (
                    ratio < getattr(self.config, 'ocr_alnum_ratio_min', 0.0)
                )
                return {
                    'length': length,
                    'alnum_ratio': ratio,
                    'min_required_length': min_required,
                    'needs_ocr': needs_ocr,
                }

            work_plan: List[Dict[str, Any]] = []
            for idx, page in enumerate(pdf.pages):
                embedded = page.extract_text() or ""
                embedded_norm = "\n".join([ln.strip() for ln in embedded.splitlines()]).strip()
                stats = analyze_text_quality(embedded_norm)
                use_ocr = self.config.ocr_enabled and stats.get('needs_ocr', False)
                work_plan.append({
                    'idx': idx,
                    'embedded_text': embedded_norm,
                    'use_ocr': use_ocr,
                    'text_stats': stats,
                })

            # Compute adaptive worker count
            cpu_count = multiprocessing.cpu_count() or 1
            max_by_cpu = cpu_count
            max_by_mem = cpu_count
            try:
                if psutil:
                    avail_mb = int(psutil.virtual_memory().available / (1024 * 1024))
                    max_by_mem = max(1, avail_mb // max(1, self.config.worker_mem_estimate_mb))
            except Exception:
                max_by_mem = cpu_count

            auto_workers = max(1, min(max_by_cpu, max_by_mem))
            configured = self.config.max_workers if getattr(self.config, 'max_workers', 0) else auto_workers
            max_workers = min(auto_workers, configured) if configured > 0 else auto_workers
            if not getattr(self.config, 'enable_parallel', True):
                max_workers = 1

            # Soft limiter: observe current CPU and available memory and reduce workers if host is busy
            try:
                if psutil:
                    cpu_pct = psutil.cpu_percent(interval=0.25)
                    avail_mb = int(psutil.virtual_memory().available / (1024 * 1024))
                    resource_snapshot['cpu_percent'] = cpu_pct
                    resource_snapshot['available_ram_mb'] = avail_mb
                    if getattr(self.config, 'log_runtime_metrics', False):
                        logger.info(f"Runtime resources — CPU: {cpu_pct}%, Available RAM: {avail_mb}MB")
                    # If CPU is high, be conservative
                    if cpu_pct and cpu_pct > getattr(self.config, 'cpu_soft_limit_percent', 80):
                        reduced = max(1, int(max_workers * 0.5))
                        logger.info(f"High CPU ({cpu_pct}%), reducing workers {max_workers} -> {reduced}")
                        max_workers = reduced
                    # If available memory is low relative to estimate, reduce workers
                    try:
                        est_needed = max_workers * max(1, getattr(self.config, 'worker_mem_estimate_mb', 512))
                        if avail_mb and avail_mb < est_needed:
                            new_by_mem = max(1, avail_mb // max(1, getattr(self.config, 'worker_mem_estimate_mb', 512)))
                            if new_by_mem < max_workers:
                                logger.info(f"Low memory ({avail_mb}MB), reducing workers {max_workers} -> {new_by_mem}")
                                max_workers = new_by_mem
                    except Exception:
                        pass
            except Exception:
                pass

            logger.debug(f"Parallel workers: {max_workers} (cpu={cpu_count}, auto={auto_workers})")

            # Worker function
            def _process_page_item(item: Dict[str, Any]) -> Dict[str, Any]:
                idx = item['idx']
                page_num = idx + 1
                page_start_time = time.time()
                embedded_text = item['embedded_text']
                use_ocr = item['use_ocr']
                text_stats = item.get('text_stats') or {}
                page_render_path: Optional[str] = None
                page_render_rel: Optional[str] = None
                produced_files_local: List[str] = []

                def ensure_page_rendered() -> Optional[str]:
                    nonlocal page_render_path, page_render_rel
                    if page_render_path:
                        return page_render_path
                    try:
                        page_render_path = render_page_to_image(pdf_path, idx, self.config.dpi, images_dir)
                        produced_files_local.append(page_render_path)
                        page_render_rel = os.path.relpath(page_render_path, base_dir).replace('\\', '/')
                    except Exception as e:
                        logger.debug(f"Render failed for page {page_num}: {e}")
                        page_render_path = None
                        page_render_rel = None
                    return page_render_path

                image_files: List[str] = []
                if self.config.preserve_images:
                    if ensure_page_rendered() and page_render_rel:
                        image_files.append(page_render_rel)

                embedded_images: List[Dict[str, Any]] = []
                embeds = []
                if self.config.preserve_images:
                    try:
                        embeds = extract_embedded_images(pdf_path, idx, extracted_dir)
                        for p in embeds:
                            produced_files_local.append(p)
                        rels = [os.path.relpath(p, base_dir).replace('\\', '/') for p in embeds]
                        image_files.extend(rels)
                        if self.config.ocr_enabled and self.config.enable_embedded_image_ocr:
                                # Optionally OCR embedded images in parallel (bounded per page)
                                try:
                                    ensure_tesseract_available()
                                    per_page_workers = max(1, getattr(self.config, 'per_page_max_workers', 1))
                                    def _ocr_emb(p_local, rel_local):
                                        try:
                                            img_e = Image.open(p_local)
                                            pre_e = preprocess_image(img_e)
                                            ocr_e = ocr_image_with_layout(pre_e, lang=self.config.lang)
                                            text_e = (ocr_e.get('text') or '').strip() if isinstance(ocr_e, dict) else (ocr_e or '')
                                            return {'path': rel_local, 'text': text_e, 'ocr_data': ocr_e.get('data') if isinstance(ocr_e, dict) else None}
                                        except Exception as e:
                                            logger.debug(f"Embedded image OCR failed for {p_local}: {e}")
                                            return {'path': rel_local, 'text': '', 'ocr_data': None}

                                    if len(embeds) <= 1 or per_page_workers <= 1:
                                        # small pages — do sequentially
                                        for p, rel in zip(embeds, rels):
                                            embedded_images.append(_ocr_emb(p, rel))
                                    else:
                                        # run small ThreadPool per-page to OCR embedded images concurrently
                                        with concurrent.futures.ThreadPoolExecutor(max_workers=min(per_page_workers, len(embeds))) as img_ex:
                                            futs = [img_ex.submit(_ocr_emb, p, rel) for p, rel in zip(embeds, rels)]
                                            for f in concurrent.futures.as_completed(futs):
                                                try:
                                                    embedded_images.append(f.result())
                                                except Exception:
                                                    embedded_images.append({'path': '', 'text': '', 'ocr_data': None})
                                except Exception as e:
                                    logger.debug(f"Embedded image OCR (parallel) failed for page {page_num}: {e}")
                        else:
                            embedded_images = [{'path': rel, 'text': '', 'ocr_data': None} for rel in rels]
                    except Exception as e:
                        logger.debug(f"Embedded images extraction failed for page {page_num}: {e}")

                diagrams_found: List[Dict[str, Any]] = []
                if self.config.enable_diagram_detection:
                    rendered_for_diagram = ensure_page_rendered()
                    try:
                        if rendered_for_diagram:
                            if page_render_rel and page_render_rel not in image_files:
                                image_files.append(page_render_rel)
                            page_diags = detect_diagrams(rendered_for_diagram, ocr_lang=self.config.lang)
                        else:
                            page_diags = []
                        if page_diags:
                            diagrams_found.extend(page_diags)
                        if self.config.preserve_images:
                            for p in embeds:
                                try:
                                    ed = detect_diagrams(p, ocr_lang=self.config.lang)
                                    if ed:
                                        diagrams_found.extend(ed)
                                except Exception:
                                    continue
                    except Exception as e:
                        logger.debug(f"Diagram detection error page {page_num}: {e}")

                ocr_text = ""
                text_source = "embedded"
                text_value = embedded_text
                ocr_blocks = None
                if use_ocr:
                    try:
                        ensure_tesseract_available()
                        rendered = ensure_page_rendered()
                        if not rendered:
                            raise RuntimeError("Unable to render page for OCR")
                        if page_render_rel and page_render_rel not in image_files:
                            image_files.insert(0, page_render_rel)
                        img = Image.open(rendered)
                        pre = preprocess_image(img)
                        ocr = ocr_image_with_layout(pre, lang=self.config.lang)
                        ocr_text = (ocr.get('text') or '').strip()
                        text_source = 'ocr'
                        text_value = ocr_text
                        ocr_blocks = ocr.get('data')
                    except Exception as e:
                        logger.debug(f"OCR failed for page {page_num}: {e}")

                extraction_time = int((time.time() - page_start_time) * 1000)
                source_details = {
                    'embedded_text_length': text_stats.get('length', len(embedded_text)),
                    'embedded_text_alnum_ratio': text_stats.get('alnum_ratio'),
                    'min_required_length': text_stats.get('min_required_length', self.config.embedded_text_threshold),
                    'heuristic_needs_ocr': text_stats.get('needs_ocr', False),
                    'ocr_used': use_ocr,
                    'threshold': text_stats.get('min_required_length', self.config.embedded_text_threshold),
                    'page_rendered': page_render_path is not None,
                    'extraction_time_ms': extraction_time,
                    'text_source': text_source,
                }

                result = {
                    'page': PageResult(
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
                    ),
                    'produced': produced_files_local,
                }
                return result

            # Execute work plan with ThreadPoolExecutor to avoid pickling issues and keep memory under control
            results_by_page: Dict[int, PageResult] = {}
            produced_by_page: Dict[int, List[str]] = {}

            if max_workers <= 1:
                for item in work_plan:
                    res = _process_page_item(item)
                    pg = res['page']
                    results_by_page[pg.page_number] = pg
                    produced_by_page[pg.page_number] = res.get('produced', [])
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = {ex.submit(_process_page_item, item): item for item in work_plan}
                    for fut in concurrent.futures.as_completed(futures):
                        item = futures.get(fut)
                        try:
                            res = fut.result()
                            pg = res['page']
                            results_by_page[pg.page_number] = pg
                            produced_by_page[pg.page_number] = res.get('produced', [])
                        except Exception as e:
                            logger.debug(f"Page processing failed for item {item}: {e}")

            # Reconstruct ordered pages and produced files by ascending page number
            for page_num in sorted(results_by_page.keys()):
                pages.append(results_by_page[page_num])
                produced.extend(produced_by_page.get(page_num, []))

        # Exporters
        produced_files: List[str] = []
        wants_txt = self.config.wants("txt")
        textexp: Optional[TextExporter] = None
        if wants_txt or self.config.enable_per_page_text_exports:
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
            if textexp and self.config.enable_per_page_text_exports:
                produced_files.append(textexp.write_page(p.page_number, page_text_combined))
        if textexp and wants_txt:
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

        runtime_stats = {
            'elapsed_ms': int((time.time() - doc_start_time) * 1000),
            'cpu_percent': resource_snapshot.get('cpu_percent'),
            'available_ram_mb': resource_snapshot.get('available_ram_mb'),
            'max_page_workers': max_workers,
            'parallel_enabled': max_workers > 1,
            'pipeline_mode': getattr(self.config, 'pipeline_mode', 'full'),
        }

        return DocumentResult(
            filename=filename,
            input_path=pdf_path,
            page_count=page_count,
            metadata=metadata,
            pages=pages,
            produced_files=produced_files,
            runtime_stats=runtime_stats,
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
