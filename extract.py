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

# Download NLTK resources
nltk.download('punkt')

# CONFIGURATION
INPUT_DIR = "resources"  # Directory containing PDFs
OUTPUT_DIR = "processed_data"
CHUNK_SIZE = 512  # Max token chunking size for fine-tuning
QA_GENERATION = True  # Enable Q&A extraction
SUMMARY_GENERATION = True  # Enable summary generation
RAG_FORMAT = True  # Enable embedding-ready chunking
EXPORT_HF = False  # Auto-upload dataset to Hugging Face (set your repo)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---

def preprocess_image(img_pil):
    """Preprocesses an image to improve OCR accuracy."""
    img = np.array(img_pil.convert("L"))  # Convert to grayscale
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Binarization
    return Image.fromarray(img)

def extract_text_from_pdf(pdf_path):
    """Extracts structured text from a PDF using PyMuPDF with OCR fallback."""
    try:
        doc = fitz.open(pdf_path)
        full_text = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")  # Extract structured text

            if not text.strip():  # If the page is empty, attempt OCR
                print(f"⚠️ OCR required for page {page_num + 1} in {pdf_path}")
                img = page.get_pixmap()
                img_pil = Image.frombytes("RGB", [img.width, img.height], img.samples)

                # Preprocess and run OCR
                text = pytesseract.image_to_string(preprocess_image(img_pil), lang="eng")

                # Save debug images and extracted text for manual verification
                img_pil.save(f"{OUTPUT_DIR}/debug_page_{page_num+1}.png")
                with open(f"{OUTPUT_DIR}/debug_page_{page_num+1}.txt", "w", encoding="utf-8") as f:
                    f.write(text)

            if text and len(text.strip()) > 10:  # Ensure non-trivial text
                full_text.append(text.strip())

        if not full_text:
            print(f"⚠️ No extractable text found in {pdf_path}. Skipping...")
            return ""

        return "\n".join(full_text)

    except Exception as e:
        print(f"❌ Error extracting text from {pdf_path}: {str(e)}")
        return ""

def clean_text(text):
    """Removes artifacts, normalizes encoding, and cleans extracted text."""
    text = unicodedata.normalize("NFKD", text)  # Fix Unicode issues
    text = unidecode(text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)  # Remove dates
    text = re.sub(r'Page \d+|page \d+', '', text)  # Remove page numbers
    text = re.sub(r'\b(?:table of contents|introduction|conclusion)\b', '', text, flags=re.I)  # Remove unnecessary sections
    return text

def split_into_chunks(text, chunk_size=CHUNK_SIZE):
    """Splits text into smaller chunks for RAG or fine-tuning."""
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def generate_qa_pairs(text_chunks):
    """Creates simple Q&A pairs from text chunks."""
    qa_pairs = []
    for chunk in text_chunks:
        question = f"What is discussed in this section?"
        answer = chunk
        qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs

def generate_summary(text_chunks):
    """Generates basic summaries using heuristic methods (no LLM required)."""
    summaries = []
    for chunk in text_chunks:
        first_sentence = nltk.sent_tokenize(chunk)[0]  # Take the first sentence as a summary
        summaries.append({"original_text": chunk, "summary": first_sentence})
    return summaries

def process_pdf(pdf_path):
    """Processes a single PDF: extract, clean, chunk, and format text."""
    try:
        text = extract_text_from_pdf(pdf_path)

        if not text or len(text.strip()) == 0:
            print(f"⚠️ Warning: No text found in {pdf_path}. Skipping...")
            return None  # Skip this PDF

        text = clean_text(text)
        text_chunks = split_into_chunks(text)

        if not text_chunks:
            print(f"⚠️ Warning: No valid chunks extracted from {pdf_path}. Skipping...")
            return None

        output = {"filename": os.path.basename(pdf_path), "chunks": text_chunks}

        if QA_GENERATION:
            output["qa_pairs"] = generate_qa_pairs(text_chunks)

        if SUMMARY_GENERATION:
            output["summaries"] = generate_summary(text_chunks)

        return output

    except Exception as e:
        print(f"❌ Unexpected error in {pdf_path}: {str(e)}")
        return None

# --- Main Processing Loop ---

def process_pdfs(input_dir=INPUT_DIR):
    """Processes all PDFs and saves structured output for AI fine-tuning & RAG."""
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    all_data = []
    failed_pdfs = []  # Track problematic PDFs

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(input_dir, pdf_file)
        processed_data = process_pdf(pdf_path)

        if processed_data:
            all_data.append(processed_data)
        else:
            failed_pdfs.append(pdf_file)

    # Save log of failed PDFs
    if failed_pdfs:
        with open(os.path.join(OUTPUT_DIR, "failed_pdfs.txt"), "w") as f:
            f.write("\n".join(failed_pdfs))
        print(f"⚠️ Logged {len(failed_pdfs)} problematic PDFs in 'failed_pdfs.txt'.")

    # Save structured data
    json_output_path = os.path.join(OUTPUT_DIR, "processed_data.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4)

    print(f"✅ Processed {len(pdf_files)} PDFs. Output saved to {OUTPUT_DIR}.")

# Run processing
if __name__ == "__main__":
    process_pdfs()
