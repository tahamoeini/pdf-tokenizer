import os
import json
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
import pandas as pd
import unicodedata
import nltk
import re
import cv2
import numpy as np
from langdetect import detect
from PIL import Image
from tqdm import tqdm
from unidecode import unidecode
from datasets import Dataset  # Hugging Face Dataset format

# --- NLTK Resource Checks ---
# Download 'punkt' if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Download 'punkt_tab' if missing
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --- CONFIGURATION ---
INPUT_DIR = "resources"      # Directory containing PDFs
OUTPUT_DIR = "processed_data"
CHUNK_SIZE = 512             # Max token chunking size for fine-tuning
QA_GENERATION = True         # Enable Q&A extraction
SUMMARY_GENERATION = True    # Enable summary generation
RAG_FORMAT = True            # Enable embedding-ready chunking
EXPORT_HF = False            # Auto-upload dataset to Hugging Face (set your repo)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---

def preprocess_image(img_pil):
    """
    Preprocesses an image (grayscale, binarization) to improve OCR accuracy.
    """
    img = np.array(img_pil.convert("L"))  # Convert to grayscale
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(img)

def extract_text_from_pdf(pdf_path):
    """
    Extracts structured text from a PDF using PyMuPDF. Falls back to OCR
    for pages with no extractable text.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")

            # If no text, use OCR
            if not text.strip():
                print(f"⚠️ OCR required for page {page_num + 1} in {pdf_path}")
                pixmap = page.get_pixmap()
                img_pil = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

                # Run OCR on the preprocessed image
                text = pytesseract.image_to_string(
                    preprocess_image(img_pil),
                    lang="eng"  # Adjust if multi-language needed
                )

                # Optional debug: save OCR images/text
                img_pil.save(os.path.join(OUTPUT_DIR, f"debug_{os.path.basename(pdf_path)}_page_{page_num + 1}.png"))
                with open(os.path.join(OUTPUT_DIR, f"debug_{os.path.basename(pdf_path)}_page_{page_num + 1}.txt"), "w", encoding="utf-8") as f:
                    f.write(text)

            # Check if extracted text is non-trivial
            if text and len(text.strip()) > 10:
                full_text.append(text.strip())

        if not full_text:
            print(f"⚠️ No extractable text found in {pdf_path}. Skipping...")
            return ""

        return "\n".join(full_text)

    except Exception as e:
        print(f"❌ Error extracting text from {pdf_path}: {str(e)}")
        return ""

def clean_text(text):
    """
    Removes artifacts, normalizes encoding, and cleans extracted text.
    """
    # Fix Unicode issues
    text = unicodedata.normalize("NFKD", text)
    # Remove non-ASCII characters
    text = unidecode(text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove dates (common format: mm/dd/yyyy)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)
    # Remove page numbers
    text = re.sub(r'Page \d+|page \d+', '', text)
    # Remove common boilerplate terms
    text = re.sub(r'\b(?:table of contents|introduction|conclusion)\b', '', text, flags=re.I)
    return text

def split_into_chunks(text, chunk_size=CHUNK_SIZE):
    """
    Splits text into smaller chunks based on sentences. 
    Ensures each chunk is below 'chunk_size' characters.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Filter out empty or near-empty chunks
    chunks = [c for c in chunks if len(c.strip()) > 0]

    return chunks

def generate_qa_pairs(text_chunks):
    """
    Creates basic Q&A pairs from text chunks.
    """
    qa_pairs = []
    for chunk in text_chunks:
        question = "What is discussed in this section?"
        answer = chunk
        qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs

def generate_summary(text_chunks):
    """
    Generates a simple summary by taking the first sentence from each chunk.
    """
    summaries = []
    for chunk in text_chunks:
        sentences = nltk.sent_tokenize(chunk)
        if not sentences:
            # Skip or handle empty chunk
            summaries.append({"original_text": chunk, "summary": ""})
            continue
        first_sentence = sentences[0]
        summaries.append({"original_text": chunk, "summary": first_sentence})
    return summaries

def process_pdf(pdf_path):
    """
    Full pipeline for a single PDF:
    1. Extract text (or OCR fallback)
    2. Clean text
    3. Chunk text
    4. Generate optional QA pairs & summaries
    """
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"⚠️ Warning: No text found in {pdf_path}. Skipping...")
            return None

        text = clean_text(text)
        text_chunks = split_into_chunks(text)

        if not text_chunks:
            print(f"⚠️ Warning: No valid chunks extracted from {pdf_path}. Skipping...")
            return None

        # Prepare final output data
        output = {"filename": os.path.basename(pdf_path), "chunks": text_chunks}

        if QA_GENERATION:
            output["qa_pairs"] = generate_qa_pairs(text_chunks)

        if SUMMARY_GENERATION:
            output["summaries"] = generate_summary(text_chunks)

        return output

    except Exception as e:
        print(f"❌ Unexpected error in {pdf_path}: {str(e)}")
        return None

def process_pdfs(input_dir=INPUT_DIR):
    """
    Loops over all PDFs in 'input_dir' and applies 'process_pdf' to each.
    Saves extracted data to JSON and logs failures in 'failed_pdfs.txt'.
    """
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    all_data = []
    failed_pdfs = []

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(input_dir, pdf_file)
        processed_data = process_pdf(pdf_path)

        if processed_data:
            all_data.append(processed_data)
        else:
            failed_pdfs.append(pdf_file)

    # Log problematic PDFs
    if failed_pdfs:
        with open(os.path.join(OUTPUT_DIR, "failed_pdfs.txt"), "w") as f:
            f.write("\n".join(failed_pdfs))
        print(f"⚠️ Logged {len(failed_pdfs)} problematic PDFs in 'failed_pdfs.txt'.")

    # Save structured data
    output_json = os.path.join(OUTPUT_DIR, "processed_data.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4)

    print(f"✅ Processed {len(pdf_files)} PDFs. Output saved to {OUTPUT_DIR}.")

if __name__ == "__main__":
    process_pdfs()
