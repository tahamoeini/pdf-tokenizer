# PDF Tokenizer & Structure Extractor

**A powerful PDF extraction tool with GUI and CLI support.**

## Features

✨ **Full Structure Preservation**
- Extracts text, tables, lists, and headings
- Preserves document structure and formatting
- Converts images/charts/diagrams to text via OCR

🎨 **GUI Application (PyQt5)**
- Drag-and-drop PDF files
- Select custom output folder
- Real-time progress and logging
- One-click output folder access

⚙️ **Flexible Output Formats**
- **JSON**: Chunked data with Q&A pairs and summaries (default)
- **Markdown**: Full-structure preservation (can be toggled in `extract.py`)

📦 **Exportable as Standalone EXE**
- Build Windows executable with PyInstaller
- No Python installation required for end users

## Quick Start

### Prerequisites
- Python 3.8+
- Tesseract OCR (download from: https://github.com/UB-Mannheim/tesseract/wiki)

### Installation

1. **Clone and setup environment** (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

2. **Run the GUI**:
```powershell
python gui.py
```

### Usage (GUI)

1. **Drag-and-drop** PDF files into the window OR click **Browse PDFs** to select files
2. (Optional) Click **Select Output Folder** to choose a custom output location
3. Click **▶ Extract PDFs** to start processing
4. View progress and logs in real-time
5. Click **📂 Open Output Folder** to access results

### Usage (CLI)

1. Place PDFs in the `resources/` folder
2. Edit `extract.py` if needed (e.g., `OUTPUT_FORMAT`, `PRESERVE_IMAGES`)
3. Run:
```powershell
python extract.py
```

## Building the EXE

To create a standalone Windows executable (no Python required):

```powershell
# Install PyInstaller (already in requirements.txt)
pip install PyInstaller

# Build exe
python build_exe.py
```

The exe will be in the `dist/` folder: `PDFTokenizer.exe`

Users can then drag-and-drop PDFs directly into the exe without needing Python installed.

## Output

### Default (JSON format)
- **processed_data.json**: Structured data with chunks, metadata, Q&A pairs, summaries
- **extracted_images/**: All extracted images (PNG)
- **processing_stats.json**: Statistics and metadata
- **processing_YYYYMMDD_HHMMSS.log**: Detailed logs

### Markdown format
- **<filename>.md**: Full markdown document with structure preserved
- **extracted_images/**: All extracted images

## Configuration

Edit `extract.py` to customize:

```python
OUTPUT_FORMAT = "json"              # or "markdown"
PRESERVE_IMAGES = True              # Extract and OCR images
EXTRACT_METADATA = True             # Extract PDF metadata
CHUNK_SIZE = 512                    # Chunk size in characters
TOKEN_LIMIT = 512                   # Max tokens per chunk
```

## Troubleshooting

**"Tesseract OCR not found"**
- Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
- Or set path in `extract.py`:
  ```python
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

**"ModuleNotFoundError" when running GUI**
- Ensure you activated the venv: `.\.venv\Scripts\Activate.ps1`
- Reinstall deps: `pip install -r requirements.txt`

**Poor OCR results**
- Try installing `opencv-python` instead of `opencv-python-headless`
- Increase image zoom in `extract_images_and_charts()` function

## Git Workflow

```bash
# Initialize repo
git init
git add .
git commit -m "Initial commit: PDF tokenizer with GUI and CLI"

# Create feature branch for improvements
git checkout -b feature/improve-ocr
# ... make changes ...
git commit -am "Improve OCR preprocessing"
git push origin feature/improve-ocr
```

## Requirements

See `requirements.txt` for all dependencies. Main packages:
- PyMuPDF (fitz)
- pdfplumber
- pytesseract (+ Tesseract binary)
- PyQt5 (for GUI)
- PyInstaller (for exe generation)
- NLTK, OpenCV, Pillow

## License

This project is open source. Feel free to modify and distribute.
