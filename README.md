# PDF Tokenizer & Structure Extractor

This project extracts PDF content and preserves structure and visuals, exporting to Markdown by default.

Quick setup & run (Windows PowerShell):

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3. Install Tesseract OCR (external)

- Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
- If not on PATH, set `pytesseract.pytesseract.tesseract_cmd` in `extract.py` to the tesseract.exe path.

4. Place PDFs in the `resources/` folder and run:

```powershell
python extract.py
```

Outputs are saved to `processed_data/`:
- Markdown files (`.md`) per PDF
- Extracted images: `processed_data/extracted_images/<pdfname>/...`
- Logs and stats: `processed_data/processing_*.log`, `processing_stats.json`

Troubleshooting:
- If `ModuleNotFoundError` occurs, ensure you installed packages into the activated venv.
- If OCR doesn't extract well, try installing `opencv-python` (full) instead of `opencv-python-headless`.
- To fetch NLTK resources manually:

```powershell
python -c "import nltk; nltk.download('punkt')"
```

If you'd like, I can run the installs and execute a smoke test on any sample PDF you provide.
