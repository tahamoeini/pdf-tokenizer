import os
import sys
import json
from pathlib import Path

# Adjust path
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import extract
from config import Config


def test_source_tracking():
    """Test that source tracking is properly recorded."""
    # Create sample PDF
    sample_pdf = BASE / 'resources' / 'sample.pdf'
    if not sample_pdf.exists():
        import create_sample_pdf  # noqa: F401
    assert sample_pdf.exists(), 'Sample PDF was not created'

    # Process with default config (should use embedded text)
    res = extract.process_pdf(str(sample_pdf))
    assert res, 'Processing returned None'

    # Determine doc dir
    out_root = Path(extract.OUTPUT_DIR)
    doc_dir = out_root / 'sample'
    
    # Load JSON and verify source tracking
    json_file = doc_dir / 'output.json'
    assert json_file.exists(), 'output.json missing'
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("JSON Structure Verification:")
    print("-" * 50)
    
    for page in data.get('pages', []):
        page_num = page.get('page_number')
        text_source = page.get('text_source')
        embedded_len = len(page.get('embedded_text', ''))
        ocr_len = len(page.get('ocr_text', ''))
        source_details = page.get('source_details', {})
        extraction_time = page.get('extraction_time_ms', 0)
        
        print(f"\nPage {page_num}:")
        print(f"  Text Source: {text_source.upper()}")
        print(f"  Embedded Text Length: {embedded_len} chars")
        print(f"  OCR Text Length: {ocr_len} chars")
        print(f"  OCR Used: {source_details.get('ocr_used', False)}")
        print(f"  Threshold: {source_details.get('threshold', 'N/A')} chars")
        print(f"  Page Rendered: {source_details.get('page_rendered', False)}")
        print(f"  Extraction Time: {extraction_time}ms")
        
        # Verify fields exist
        assert 'embedded_text' in page, 'embedded_text field missing'
        assert 'ocr_text' in page, 'ocr_text field missing'
        assert 'source_details' in page, 'source_details field missing'
        assert 'extraction_time_ms' in page, 'extraction_time_ms field missing'
        
        # Verify source_details structure
        assert 'embedded_text_length' in source_details
        assert 'ocr_used' in source_details
        assert 'threshold' in source_details
        assert 'page_rendered' in source_details
        assert 'extraction_time_ms' in source_details
        assert 'text_source' in source_details

    # Check Markdown for source headers
    md_file = doc_dir / 'output.md'
    assert md_file.exists(), 'output.md missing'
    
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    print("\nMarkdown Structure Verification:")
    print("-" * 50)
    
    # Verify markdown contains expected headers
    assert '**Extraction Source:**' in md_content, 'Missing source indicator'
    assert '### 📄 Extracted Text' in md_content, 'Missing embedded text header'
    assert 'Extraction Time:' in md_content, 'Missing timing info'
    
    print("[OK] Markdown contains source indicator")
    print("[OK] Markdown contains embedded text header")
    print("[OK] Markdown contains timing info")
    
    # Verify images and diagrams sections
    if '### 🖼️ Images in Page' in md_content:
        print("[OK] Markdown contains images section")
    if '### 📊 Detected Diagrams' in md_content:
        print("[OK] Markdown contains diagrams section")
    
    print("\n" + "=" * 50)
    print("SUCCESS: All source tracking tests passed!")
    print(f"Outputs at: {doc_dir}")
    

if __name__ == '__main__':
    test_source_tracking()
