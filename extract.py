import os
import json
import sys
import logging
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
import unicodedata
import nltk
import re
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from unidecode import unidecode
from datetime import datetime
import shutil
import warnings
import io
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

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
def check_dependencies():
    """Verify all required dependencies are installed and accessible."""
    logger = logging.getLogger(__name__)
    issues = []
    
    # Check Tesseract OCR
    try:
        pytesseract.get_tesseract_version()
        logger.info("✅ Tesseract OCR found")
    except pytesseract.TesseractNotFoundError:
        issues.append("❌ Tesseract OCR not found. Install from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    # Check NLTK resources
    for resource in ['tokenizers/punkt', 'tokenizers/punkt_tab']:
        try:
            nltk.data.find(resource)
            logger.info(f"✅ NLTK resource '{resource}' found")
        except LookupError:
            logger.info(f"📥 Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)
    
    # Check tiktoken for token counting
    try:
        import tiktoken
        logger.info("✅ tiktoken available for token-based chunking")
    except ImportError:
        logger.warning("⚠️ tiktoken not installed. Token-based chunking will use estimate.")
    
    if issues:
        logger.error(f"Dependency check failed:\n" + "\n".join(issues))
        return False
    
    return True

# --- CONFIGURATION & VALIDATION ---
INPUT_DIR = "resources"      # Directory containing PDFs
OUTPUT_DIR = "processed_data"
CHUNK_SIZE = 512             # Target chunk size (characters)
TOKEN_LIMIT = 512            # Max tokens per chunk
QA_GENERATION = True         # Enable Q&A extraction (JSON mode only)
SUMMARY_GENERATION = True    # Enable summary generation (JSON mode only)
EXTRACT_METADATA = True      # Enable PDF metadata extraction
OUTPUT_FORMAT = "json"   # Output format: "markdown" or "json"
PRESERVE_IMAGES = True       # Extract and convert images to text
PRESERVE_STRUCTURE = True    # Preserve document structure
MIN_CHUNK_LENGTH = 50        # Minimum characters per valid chunk
IMAGES_SUBDIR = "extracted_images"

def validate_config():
    """Validate configuration at startup."""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(INPUT_DIR):
        logger.error(f"❌ Input directory '{INPUT_DIR}' not found!")
        return False
    
    if OUTPUT_FORMAT not in ["markdown", "json"]:
        logger.error(f"❌ Invalid OUTPUT_FORMAT. Use 'markdown' or 'json'")
        return False
    
    logger.info(f"✅ Configuration validated (Output: {OUTPUT_FORMAT})")
    return True

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize logging
logger = setup_logging(OUTPUT_DIR)

# --- Helper Functions ---

def estimate_tokens(text):
    """Estimate token count (1 token ≈ 4 characters for English)."""
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except (ImportError, KeyError):
        return len(text) // 4

def preprocess_image(img_pil):
    """Preprocesses an image for better OCR accuracy."""
    try:
        # Convert to grayscale and upscale for better OCR
        img = np.array(img_pil.convert("L"))
        h, w = img.shape[:2]
        scale = 2 if max(w, h) < 2000 else 1
        if scale != 1:
            img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # Denoise and enhance contrast
        img = cv2.fastNlMeansDenoising(img, None, h=10)
        img = cv2.equalizeHist(img)

        # Adaptive thresholding for varying backgrounds
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 35, 15)

        # Morphological closing to connect broken strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        return Image.fromarray(img)
    except Exception as e:
        logger.warning(f"⚠️ Image preprocessing failed: {e}")
        return img_pil


def ocr_image_with_layout(img_pil, lang='eng'):
    """Run OCR and return text using layout-aware parsing (word->line grouping).
    Returns a cleaned textual representation that preserves layout order.
    """
    try:
        # Use image_to_data to get word-level boxes and line numbers
        data = pytesseract.image_to_data(img_pil, lang=lang, output_type=pytesseract.Output.DICT)
        n_boxes = len(data.get('level', []))
        lines = {}
        for i in range(n_boxes):
            block = data['block_num'][i]
            par = data['par_num'][i]
            line = data['line_num'][i]
            text = data['text'][i].strip()
            if not text:
                continue
            key = (block, par, line)
            lines.setdefault(key, []).append(text)

        ordered = []
        for key in sorted(lines.keys()):
            ordered.append(' '.join(lines[key]))

        # Fallback to plain OCR string if nothing found
        raw = pytesseract.image_to_string(img_pil, lang=lang)
        text = '\n'.join(ordered) if ordered else raw
        return text.strip()
    except Exception as e:
        logger.debug(f"⚠️ OCR layout parsing failed: {e}")
        try:
            return pytesseract.image_to_string(img_pil, lang=lang)
        except Exception:
            return ""

def extract_pdf_metadata(pdf_path):
    """Extracts metadata from PDF."""
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata or {}
        page_count = len(doc)
        doc.close()
        
        return {
            "title": metadata.get("title", "Unknown"),
            "author": metadata.get("author", "Unknown"),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "page_count": page_count
        }
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract metadata: {e}")
        return {}

def extract_images_and_charts(pdf_path, page_num, output_subdir):
    """Extracts images from a PDF page and converts to text."""
    images_data = []
    
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                if pix.n - pix.alpha < 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                img_path = os.path.join(output_subdir, f"page_{page_num + 1}_img_{img_index + 1}.png")
                pix.save(img_path)
                
                # Extract text from image using improved OCR + layout parsing
                img_pil = Image.open(img_path)
                pre = preprocess_image(img_pil)
                img_text = ocr_image_with_layout(pre, lang='eng')
                
                if not img_text.strip():
                    # As a fallback, try plain OCR again
                    img_text = pytesseract.image_to_string(pre, lang='eng')

                images_data.append({
                    "type": "image",
                    "page": page_num + 1,
                    "index": img_index + 1,
                    "path": img_path,
                    "extracted_text": img_text.strip() if img_text.strip() else "[Visual content - diagram/chart/image]",
                    "markdown_ref": f"![Image p{page_num + 1}](extracted_images/page_{page_num + 1}_img_{img_index + 1}.png)"
                })
                
                logger.debug(f"📸 Extracted image from page {page_num + 1}")
                
            except Exception as e:
                logger.debug(f"⚠️ Failed to extract image {img_index}: {e}")
                continue
        
        doc.close()
        return images_data
        
    except Exception as e:
        logger.debug(f"⚠️ Error extracting images: {e}")
        return []

def extract_with_structure(pdf_path, output_subdir):
    """Extracts text with structure preservation using pdfplumber."""
    structured_content = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_content = {
                    "page": page_num + 1,
                    "elements": []
                }
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        if not table:
                            continue
                        
                        # Build markdown table
                        header = table[0]
                        table_md = "| " + " | ".join([str(cell) if cell else "" for cell in header]) + " |\n"
                        table_md += "|" + "|".join(["---"] * len(header)) + "|\n"
                        
                        for row in table[1:]:
                            table_md += "| " + " | ".join([str(cell) if cell else "" for cell in row]) + " |\n"
                        
                        page_content["elements"].append({
                            "type": "table",
                            "content": table_md.strip()
                        })
                
                # Extract text with structure detection
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        element_type = "paragraph"
                        
                        # Detect headings
                        if len(line) < 100 and (line.isupper() or re.match(r'^(Chapter|Section|Part|Title)', line, re.I)):
                            element_type = "heading"
                        
                        # Detect lists
                        if line.startswith(("•", "-", "*", "◦", "○")) or re.match(r'^(\d+\.|[a-z]\.|[A-Z]\.)', line):
                            element_type = "list_item"
                        
                        page_content["elements"].append({
                            "type": element_type,
                            "content": line
                        })
                
                # Extract images
                if PRESERVE_IMAGES:
                    images = extract_images_and_charts(pdf_path, page_num, output_subdir)
                    for img in images:
                        page_content["elements"].append({
                            "type": "image",
                            "content": img["extracted_text"],
                            "markdown_ref": img["markdown_ref"]
                        })
                
                if page_content["elements"]:
                    structured_content.append(page_content)
        
        return structured_content
        
    except Exception as e:
        logger.error(f"❌ Error extracting structured content: {e}")
        return []

def convert_to_markdown(pdf_data, structured_content):
    """Converts extracted content to markdown with full structure preservation."""
    md_content = []
    
    # Document header with metadata
    if pdf_data.get("metadata"):
        meta = pdf_data["metadata"]
        title = meta.get("title") or pdf_data.get("filename", "Document")
        md_content.append(f"# {title}\n")
        
        if meta.get("author") and meta["author"] != "Unknown":
            md_content.append(f"**Author:** {meta['author']}")
        if meta.get("subject"):
            md_content.append(f"**Subject:** {meta['subject']}")
        if meta.get("page_count"):
            md_content.append(f"**Total Pages:** {meta['page_count']}")
        
        md_content.append("\n---\n")
    
    # Process each page
    for page_content in structured_content:
        page_num = page_content["page"]
        md_content.append(f"## Page {page_num}\n")
        
        for element in page_content["elements"]:
            elem_type = element["type"]
            content = element.get("content", "").strip()
            
            if not content:
                continue
            
            if elem_type == "heading":
                md_content.append(f"### {content}\n")
                
            elif elem_type == "list_item":
                # Clean and reformat as markdown list
                cleaned = re.sub(r'^[•\-*◦○]?\s*', '', content)
                md_content.append(f"- {cleaned}")
                
            elif elem_type == "table":
                md_content.append(f"{content}\n")
                
            elif elem_type == "image":
                if element.get("markdown_ref"):
                    md_content.append(f"{element['markdown_ref']}\n")
                
                if content and "[Visual" not in content:
                    md_content.append(f"*[Image Content]*: {content}\n")
                
            elif elem_type == "paragraph":
                md_content.append(f"{content}\n")
        
        md_content.append("\n---\n")
    
    return "\n".join(md_content)

def split_into_chunks(text, chunk_size=CHUNK_SIZE, token_limit=TOKEN_LIMIT):
    """Splits text into chunks with token awareness (JSON mode only)."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        test_chunk = (current_chunk + " " + sentence).strip()
        
        if len(test_chunk) < chunk_size and estimate_tokens(test_chunk) < token_limit:
            current_chunk = test_chunk
        else:
            if len(current_chunk.strip()) >= MIN_CHUNK_LENGTH:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if len(current_chunk.strip()) >= MIN_CHUNK_LENGTH:
        chunks.append(current_chunk.strip())

    logger.debug(f"📊 Created {len(chunks)} chunks")
    return chunks

def generate_qa_pairs(text_chunks):
    """Creates Q&A pairs from chunks (JSON mode only)."""
    qa_pairs = []
    
    for chunk in text_chunks:
        sentences = nltk.sent_tokenize(chunk)
        if not sentences:
            continue
        
        qa_pairs.append({
            "question": "What is the main topic?",
            "answer": chunk
        })
    
    return qa_pairs

def generate_summary(text_chunks):
    """Generates summaries from chunks (JSON mode only)."""
    summaries = []
    
    for chunk in text_chunks:
        sentences = nltk.sent_tokenize(chunk)
        summary = sentences[0] if sentences else ""
        
        summaries.append({
            "original_text": chunk[:100] + ("..." if len(chunk) > 100 else ""),
            "summary": summary
        })
    
    return summaries

def process_pdf(pdf_path):
    """Main PDF processing pipeline."""
    try:
        filename = os.path.basename(pdf_path)
        logger.info(f"📖 Processing: {filename}")
        
        # Create subdirectory for this PDF's images
        pdf_images_dir = os.path.join(OUTPUT_DIR, IMAGES_SUBDIR, filename.replace('.pdf', ''))
        os.makedirs(pdf_images_dir, exist_ok=True)
        
        # Extract metadata
        metadata = extract_pdf_metadata(pdf_path) if EXTRACT_METADATA else {}
        
        # Extract content with structure
        structured_content = extract_with_structure(pdf_path, pdf_images_dir)
        
        if not structured_content:
            logger.warning(f"⚠️ No content extracted from {filename}")
            return None

        # Prepare output based on format
        if OUTPUT_FORMAT == "markdown":
            # Generate markdown document
            pdf_data = {
                "filename": filename,
                "metadata": metadata
            }
            md_content = convert_to_markdown(pdf_data, structured_content)
            
            # Save markdown file
            md_filename = os.path.join(OUTPUT_DIR, filename.replace('.pdf', '.md'))
            with open(md_filename, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            logger.info(f"✅ Generated markdown: {md_filename}")
            
            return {
                "filename": filename,
                "output": md_filename,
                "format": "markdown",
                "pages": len(structured_content)
            }
        
        else:  # JSON format
            # Generate chunks and Q&A
            full_text = "\n".join([
                elem.get("content", "") 
                for page in structured_content 
                for elem in page.get("elements", [])
            ])
            
            text_chunks = split_into_chunks(full_text)
            
            output = {
                "filename": filename,
                "metadata": metadata,
                "chunks": text_chunks,
                "chunk_count": len(text_chunks)
            }
            
            if QA_GENERATION:
                output["qa_pairs"] = generate_qa_pairs(text_chunks)
            
            if SUMMARY_GENERATION:
                output["summaries"] = generate_summary(text_chunks)
            
            logger.info(f"✅ Processed {filename}: {len(text_chunks)} chunks")
            return output

    except Exception as e:
        logger.error(f"❌ Error processing {pdf_path}: {e}", exc_info=True)
        return None

def process_pdfs(input_dir=INPUT_DIR):
    """Batch process all PDFs in directory."""
    logger.info("=" * 70)
    logger.info("PDF TOKENIZER & STRUCTURE EXTRACTOR - Starting Processing")
    logger.info(f"Output Format: {OUTPUT_FORMAT.upper()}")
    logger.info("=" * 70)
    
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        logger.error(f"❌ No PDF files found in '{input_dir}'")
        return
    
    logger.info(f"📁 Found {len(pdf_files)} PDF(s)\n")
    
    all_data = []
    failed_pdfs = []
    statistics = {
        "total_pdfs": len(pdf_files),
        "successful": 0,
        "failed": 0,
        "format": OUTPUT_FORMAT,
        "start_time": datetime.now().isoformat()
    }

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(input_dir, pdf_file)
        
        try:
            processed_data = process_pdf(pdf_path)
            
            if processed_data:
                all_data.append(processed_data)
                statistics["successful"] += 1
            else:
                failed_pdfs.append(pdf_file)
                statistics["failed"] += 1
        except Exception as e:
            logger.error(f"❌ Exception: {e}")
            failed_pdfs.append(pdf_file)
            statistics["failed"] += 1

    statistics["end_time"] = datetime.now().isoformat()

    # Log failures
    if failed_pdfs:
        failed_log = os.path.join(OUTPUT_DIR, "failed_pdfs.txt")
        with open(failed_log, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_pdfs))
        logger.warning(f"⚠️ {len(failed_pdfs)} failed PDFs logged")

    # Save data
    if OUTPUT_FORMAT == "json":
        output_json = os.path.join(OUTPUT_DIR, "processed_data.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Data saved to: {output_json}")

    # Save statistics
    stats_file = os.path.join(OUTPUT_DIR, "processing_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)

    # Summary
    logger.info("=" * 70)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"✅ Successful: {statistics['successful']}/{statistics['total_pdfs']}")
    logger.info(f"❌ Failed: {statistics['failed']}/{statistics['total_pdfs']}")
    logger.info(f"📂 Output: {OUTPUT_DIR}")
    logger.info("=" * 70 + "\n")

if __name__ == "__main__":
    try:
        if not check_dependencies():
            logger.error("Dependency check failed.")
            sys.exit(1)
        
        if not validate_config():
            logger.error("Configuration validation failed.")
            sys.exit(1)
        
        process_pdfs()
        logger.info("🎉 All done!")
        
    except KeyboardInterrupt:
        logger.warning("⏸️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"🔥 Critical error: {e}", exc_info=True)
        sys.exit(1)
