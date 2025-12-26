import os
import json
import sys
import logging
import pytesseract
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import warnings
import io
from typing import Optional, Dict, Any

# Suppress warnings
warnings.filterwarnings('ignore')

# Modular pipeline
from config import Config
from processor import DocumentProcessor

# --- Logging Configuration ---
def setup_logging(output_dir):
    """Initialize logging system."""
    log_file = os.path.join(output_dir, f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # File handler (UTF-8)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Stream handler with UTF-8 wrapper to avoid Windows console encoding errors
    try:
        stream = sys.stdout
        if hasattr(sys.stdout, 'buffer'):
            stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    except Exception:
        stream_handler = logging.StreamHandler(sys.stdout)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

# --- Dependency & Environment Checks ---
def check_dependencies(ocr_required: bool = True):
    """Verify required dependencies; fail fast if OCR requested but unavailable."""
    logger = logging.getLogger(__name__)
    issues = []
    
    # Check Tesseract OCR
    if ocr_required:
        try:
            pytesseract.get_tesseract_version()
            logger.info("✅ Tesseract OCR found")
        except pytesseract.TesseractNotFoundError:
            issues.append(
                "❌ Tesseract OCR not found. Install from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "   Or set pytesseract.pytesseract.tesseract_cmd to the tesseract.exe path."
            )
    
    if issues:
        logger.error(f"Dependency check failed:\n" + "\n".join(issues))
        return False
    
    return True

# --- CONFIGURATION & VALIDATION ---
INPUT_DIR = "resources"      # Directory containing PDFs
OUTPUT_DIR = "processed_data"
PRESERVE_IMAGES = True       # Extract and convert images to text

COMBINED_SEP_ARTICLE = "===== ARTICLE START ====="
COMBINED_SEP_TEXT_START = "----- TEXT START -----"
COMBINED_SEP_TEXT_END = "----- TEXT END -----"
COMBINED_SEP_ARTICLE_END = "===== ARTICLE END ====="


class CombinedWriter:
    """Handles combined JSONL/TXT aggregation across documents."""

    def __init__(self, enabled: bool, jsonl_path: Optional[str], txt_path: Optional[str]):
        self.enabled = bool(enabled)
        self.jsonl_path = jsonl_path
        self.txt_path = txt_path
        self._prepared = False

    def prepare(self):
        if not self.enabled or self._prepared:
            return
        for path in (self.jsonl_path, self.txt_path):
            if not path:
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("")
        self._prepared = True

    def append(self, summary: Optional[Dict[str, Any]]):
        if not self.enabled or not summary:
            return
        if not self._prepared:
            self.prepare()

        article_text = ""
        text_path = summary.get("full_text_path")
        if text_path and os.path.exists(text_path):
            try:
                with open(text_path, "r", encoding="utf-8") as fh:
                    article_text = fh.read()
            except Exception as exc:
                logger.warning(f"⚠️ Failed to read text for combined export ({text_path}): {exc}")
        else:
            logger.warning(f"⚠️ Missing full_text file for combined export: {text_path}")

        title = summary.get("title") or summary.get("filename")
        metadata = summary.get("metadata") or {}
        doi = metadata.get("doi")

        if self.jsonl_path:
            record = {
                "filename": summary.get("filename"),
                "title": title,
                "doi": doi,
                "pages_total": summary.get("page_count"),
                "pages_extracted": summary.get("pages_extracted"),
                "pages_ocr": summary.get("pages_ocr"),
                "embedded_images_ocr": summary.get("embedded_images_ocr"),
                "text": article_text,
            }
            with open(self.jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.txt_path:
            with open(self.txt_path, "a", encoding="utf-8") as tf:
                tf.write(
                    f"{COMBINED_SEP_ARTICLE}\n"
                    f"Title: {title}\n"
                    f"DOI: {doi or 'N/A'}\n"
                    f"Filename: {summary.get('filename')}\n"
                    f"Pages: total={summary.get('page_count')} extracted={summary.get('pages_extracted')} ocr={summary.get('pages_ocr')}\n"
                    f"EmbeddedImagesOCR: {summary.get('embedded_images_ocr')}\n"
                    f"{COMBINED_SEP_TEXT_START}\n"
                )
                tf.write(article_text + "\n")
                tf.write(f"{COMBINED_SEP_TEXT_END}\n{COMBINED_SEP_ARTICLE_END}\n\n")

def validate_config():
    """Validate configuration at startup."""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(INPUT_DIR):
        logger.error(f"❌ Input directory '{INPUT_DIR}' not found!")
        return False
    
    # OUTPUT_FORMAT retained for backward-compat; exporters are controlled internally
    logger.info(f"✅ Configuration validated (Output dir: {OUTPUT_DIR})")
    return True

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize logging
logger = setup_logging(OUTPUT_DIR)

# --- Helper Functions ---

def preprocess_image(img_pil):
    """Shim to new OCR module for backward compatibility."""
    from ocr import preprocess_image as _pre
    return _pre(img_pil)


def ocr_image_with_layout(img_pil, lang='eng'):
    """Shim to new OCR module for backward compatibility."""
    from ocr import ocr_image_with_layout as _ocr
    result = _ocr(img_pil, lang=lang)
    return result.get("text", "")

def process_pdf(
    pdf_path,
    max_workers=None,
    enable_parallel=None,
    per_page_max_workers=None,
    pipeline_mode=None,
    combined_override=None,
    combined_jsonl_path=None,
    combined_txt_path=None,
    ocr_enabled=True,
    dpi=None,
    lang=None,
    embedded_threshold=None,
):
    """Process a single PDF using the modular pipeline and write outputs."""
    try:
        filename = os.path.basename(pdf_path)
        logger.info(f"📖 Processing: {filename}")

        cfg_kwargs = dict(
            input_paths=[pdf_path],
            output_dir=OUTPUT_DIR,
            ocr_enabled=bool(ocr_enabled),
            dpi=dpi or 300,
            lang=lang or 'eng',
            embedded_text_threshold=embedded_threshold if embedded_threshold is not None else 40,
            preserve_images=PRESERVE_IMAGES,
            formats={"txt", "json", "md"},
            pipeline_mode=pipeline_mode or "full",
        )
        if max_workers is not None:
            cfg_kwargs['max_workers'] = int(max_workers)
        if enable_parallel is not None:
            cfg_kwargs['enable_parallel'] = bool(enable_parallel)
        if per_page_max_workers is not None:
            cfg_kwargs['per_page_max_workers'] = int(per_page_max_workers)
        if combined_override is not None:
            cfg_kwargs['enable_combined_exports'] = bool(combined_override)
        if combined_jsonl_path:
            cfg_kwargs['combined_jsonl_path'] = combined_jsonl_path
        if combined_txt_path:
            cfg_kwargs['combined_txt_path'] = combined_txt_path

        cfg = Config(**cfg_kwargs)

        # Defer OCR check to when OCR is actually used
        if not check_dependencies(ocr_required=cfg_kwargs['ocr_enabled']):
            raise RuntimeError("Dependencies missing. See log for details.")

        processor = DocumentProcessor(cfg)
        result = processor.process(pdf_path)

        for p in result.produced_files:
            logger.info(f"💾 Wrote: {p}")
        if result.runtime_stats:
            logger.info(f"⏱️ Runtime stats for {filename}: {result.runtime_stats}")

        pages_ocr = sum(1 for p in result.pages if p.text_source == 'ocr')
        pages_embedded = sum(1 for p in result.pages if p.text_source == 'embedded')
        embedded_images_ocr = sum(len(p.embedded_images or []) for p in result.pages)
        full_text_path = None
        for produced_path in result.produced_files:
            normalized = produced_path.replace('\\', '/').lower()
            if normalized.endswith('/text/full_text.txt'):
                full_text_path = produced_path
                break
        summary = {
            "filename": result.filename,
            "title": (result.metadata or {}).get("title") or result.filename,
            "page_count": result.page_count,
            "pages_extracted": pages_embedded,
            "pages_ocr": pages_ocr,
            "embedded_images_ocr": embedded_images_ocr,
            "full_text_path": full_text_path,
            "metadata": result.metadata,
            "runtime_stats": result.runtime_stats,
        }

        return {
            "filename": result.filename,
            "pages": result.page_count,
            "outputs": result.produced_files,
            "summary": summary,
        }

    except Exception as e:
        logger.error(f"❌ Error processing {pdf_path}: {e}", exc_info=True)
        return None

def process_pdfs(
    input_dir=INPUT_DIR,
    pipeline_mode="full",
    doc_workers=None,
    max_workers=None,
    enable_parallel=None,
    per_page_max_workers=None,
    combined_writer: Optional[CombinedWriter] = None,
    ocr_enabled=True,
    dpi=None,
    lang=None,
    embedded_threshold=None,
):
    """Batch process all PDFs in a directory with process-level parallelism."""
    logger.info("=" * 70)
    logger.info("PDF TOKENIZER & STRUCTURE EXTRACTOR - Starting Processing")
    logger.info(f"Pipeline mode: {pipeline_mode}")
    logger.info("=" * 70)

    pdf_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")])
    if not pdf_files:
        logger.error(f"❌ No PDF files found in '{input_dir}'")
        return

    logger.info(f"📁 Found {len(pdf_files)} PDF(s)\n")
    if combined_writer and combined_writer.enabled:
        logger.info(f"📝 Combined outputs -> JSONL: {combined_writer.jsonl_path}, TXT: {combined_writer.txt_path}")
    if combined_writer:
        combined_writer.prepare()

    doc_workers = doc_workers if doc_workers is not None else (os.cpu_count() or 1)
    doc_workers = max(1, min(doc_workers, len(pdf_files)))
    resolved_page_parallel = enable_parallel
    if resolved_page_parallel is None and doc_workers > 1:
        resolved_page_parallel = False

    all_data = []
    failed_pdfs = []
    statistics = {
        "total_pdfs": len(pdf_files),
        "successful": 0,
        "failed": 0,
        "formats": ["txt", "json", "md"],
        "doc_workers": doc_workers,
        "page_parallel": resolved_page_parallel if resolved_page_parallel is not None else "auto",
        "start_time": datetime.now().isoformat(),
        "documents": [],
    }

    def _handle_result(pdf_name: str, processed_data):
        if processed_data:
            all_data.append(processed_data)
            statistics["successful"] += 1
            if combined_writer:
                combined_writer.append(processed_data.get("summary"))
            summary = processed_data.get("summary") or {}
            statistics["documents"].append({
                "filename": processed_data.get("filename") or pdf_name,
                "pages": processed_data.get("pages"),
                "runtime": summary.get("runtime_stats"),
            })
        else:
            failed_pdfs.append(pdf_name)
            statistics["failed"] += 1

    progress = tqdm(total=len(pdf_files), desc="Processing PDFs", unit="file")
    if doc_workers <= 1:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_file)
            try:
                processed_data = process_pdf(
                    pdf_path,
                    max_workers=max_workers,
                    enable_parallel=resolved_page_parallel,
                    per_page_max_workers=per_page_max_workers,
                    pipeline_mode=pipeline_mode,
                    combined_override=combined_writer.enabled if combined_writer else None,
                    combined_jsonl_path=combined_writer.jsonl_path if combined_writer else None,
                    combined_txt_path=combined_writer.txt_path if combined_writer else None,
                    ocr_enabled=ocr_enabled,
                    dpi=dpi,
                    lang=lang,
                    embedded_threshold=embedded_threshold,
                )
            except Exception as exc:
                logger.error(f"❌ Exception processing {pdf_file}: {exc}")
                processed_data = None
            _handle_result(pdf_file, processed_data)
            progress.update(1)
    else:
        logger.info(f"🧵 Document workers: {doc_workers} (page parallel: {statistics['page_parallel']})")
        with ProcessPoolExecutor(max_workers=doc_workers) as pool:
            futures = {}
            for pdf_file in pdf_files:
                pdf_path = os.path.join(input_dir, pdf_file)
                task_kwargs = dict(
                    max_workers=max_workers,
                    enable_parallel=resolved_page_parallel,
                    per_page_max_workers=per_page_max_workers,
                    pipeline_mode=pipeline_mode,
                    combined_override=combined_writer.enabled if combined_writer else None,
                    combined_jsonl_path=combined_writer.jsonl_path if combined_writer else None,
                    combined_txt_path=combined_writer.txt_path if combined_writer else None,
                    ocr_enabled=ocr_enabled,
                    dpi=dpi,
                    lang=lang,
                    embedded_threshold=embedded_threshold,
                )
                futures[pool.submit(process_pdf, pdf_path, **task_kwargs)] = pdf_file
            for fut in as_completed(futures):
                pdf_file = futures[fut]
                try:
                    processed_data = fut.result()
                except Exception as exc:
                    logger.error(f"❌ Exception processing {pdf_file}: {exc}")
                    processed_data = None
                _handle_result(pdf_file, processed_data)
                progress.update(1)
    progress.close()

    statistics["end_time"] = datetime.now().isoformat()

    if failed_pdfs:
        failed_log = os.path.join(OUTPUT_DIR, "failed_pdfs.txt")
        with open(failed_log, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_pdfs))
        logger.warning(f"⚠️ {len(failed_pdfs)} failed PDFs logged")

    output_json = os.path.join(OUTPUT_DIR, "processed_data.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Run index saved to: {output_json}")

    stats_file = os.path.join(OUTPUT_DIR, "processing_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)

    logger.info("=" * 70)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"✅ Successful: {statistics['successful']}/{statistics['total_pdfs']}")
    logger.info(f"❌ Failed: {statistics['failed']}/{statistics['total_pdfs']}")
    logger.info(f"📂 Output: {OUTPUT_DIR}")
    logger.info("=" * 70 + "\n")

if __name__ == "__main__":
    import argparse
    try:
        parser = argparse.ArgumentParser(description="PDF Tokenizer & Structure Extractor")
        parser.add_argument("--input", "-i", help="PDF file or directory (defaults to resources)", default=INPUT_DIR)
        parser.add_argument("--output", "-o", help="Output directory", default=OUTPUT_DIR)
        parser.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback")
        parser.add_argument("--dpi", type=int, default=300, help="DPI for page rendering")
        parser.add_argument("--lang", type=str, default="eng", help="OCR language")
        parser.add_argument("--threshold", type=int, default=40, help="Embedded text length threshold for OCR fallback")
        parser.add_argument("--formats", type=str, default="txt,json,md", help="Comma-separated formats to generate")
        parser.add_argument("--mode", type=str, choices=["full", "fast-text"], default="full", help="Pipeline preset")
        parser.add_argument("--combined", action="store_true", help="Enable combined JSONL/TXT outputs")
        parser.add_argument("--no-combined", action="store_true", help="Disable combined JSONL/TXT outputs")
        parser.add_argument("--combined-jsonl", type=str, default=None, help="Custom path for combined JSONL output")
        parser.add_argument("--combined-txt", type=str, default=None, help="Custom path for combined TXT output")
        parser.add_argument("--max-workers", type=int, default=None, help="Per-document page workers (0=auto)")
        parser.add_argument("--doc-workers", type=int, default=None, help="Concurrent document processes")
        parser.add_argument("--per-page-workers", type=int, default=None, help="Per-page embedded-image OCR workers")
        parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
        args = parser.parse_args()

        OUTPUT_DIR = args.output
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if args.combined and args.no_combined:
            logger.error("Choose only one of --combined or --no-combined")
            sys.exit(1)

        preview_cfg = Config(input_paths=[], output_dir=OUTPUT_DIR, pipeline_mode=args.mode)
        if args.combined:
            preview_cfg.enable_combined_exports = True
        if args.no_combined:
            preview_cfg.enable_combined_exports = False
        if args.combined_jsonl:
            preview_cfg.combined_jsonl_path = preview_cfg._resolve_output_path(args.combined_jsonl)
        if args.combined_txt:
            preview_cfg.combined_txt_path = preview_cfg._resolve_output_path(args.combined_txt)

        if not validate_config():
            logger.error("Configuration validation failed.")
            sys.exit(1)

        combined_writer = CombinedWriter(
            preview_cfg.enable_combined_exports,
            preview_cfg.combined_jsonl_path if preview_cfg.enable_combined_exports else None,
            preview_cfg.combined_txt_path if preview_cfg.enable_combined_exports else None,
        )

        ocr_required = not args.no_ocr
        page_parallel_flag = False if args.no_parallel else None
        if os.path.isdir(args.input):
            if not check_dependencies(ocr_required=ocr_required):
                logger.error("Dependency check failed.")
                sys.exit(1)
            process_pdfs(
                input_dir=args.input,
                pipeline_mode=args.mode,
                doc_workers=args.doc_workers,
                max_workers=args.max_workers,
                enable_parallel=page_parallel_flag,
                per_page_max_workers=args.per_page_workers,
                combined_writer=combined_writer,
                ocr_enabled=ocr_required,
                dpi=args.dpi,
                lang=args.lang,
                embedded_threshold=args.threshold,
            )
        else:
            if not check_dependencies(ocr_required=ocr_required):
                logger.error("Dependency check failed.")
                sys.exit(1)
            res = process_pdf(args.input,
                              max_workers=args.max_workers,
                              enable_parallel=page_parallel_flag,
                              per_page_max_workers=args.per_page_workers,
                              pipeline_mode=args.mode,
                              combined_override=combined_writer.enabled if combined_writer else None,
                              combined_jsonl_path=combined_writer.jsonl_path if combined_writer else None,
                              combined_txt_path=combined_writer.txt_path if combined_writer else None,
                              ocr_enabled=ocr_required,
                              dpi=args.dpi,
                              lang=args.lang,
                              embedded_threshold=args.threshold)
            if not res:
                sys.exit(1)
            if combined_writer:
                combined_writer.append(res.get("summary"))
        logger.info("🎉 All done!")

    except KeyboardInterrupt:
        logger.warning("⏸️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"🔥 Critical error: {e}", exc_info=True)
        sys.exit(1)
