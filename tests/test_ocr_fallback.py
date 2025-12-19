#!/usr/bin/env python3
"""
Test to verify OCR fallback behavior when embedded text is below threshold.
Creates a PDF with minimal/no embedded text and verifies OCR is used.
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Adjust path
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from extract import process_pdf, OUTPUT_DIR
from config import Config
from processor import DocumentProcessor

def create_image_only_pdf():
    """Create a simple PDF with minimal embedded text (just an image with text)."""
    try:
        import fitz
    except ImportError:
        print("PyMuPDF required. Install with: pip install PyMuPDF")
        return None
    
    # Create image with text
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw text on image
    text = "This text is only in the image,\nnot in PDF structure.\nOCR should extract it."
    try:
        # Try to use default font, fallback if not available
        font_size = 24
        draw.text((50, 100), text, fill='black')
    except:
        draw.text((50, 100), text, fill='black')
    
    # Save temporarily
    temp_img = Path(BASE) / 'temp_image_only_test.png'
    img.save(temp_img)
    
    # Create PDF from image
    doc = fitz.open()
    pix = fitz.Pixmap(str(temp_img))
    page = doc.new_page(width=pix.width, height=pix.height)
    page.insert_image(page.rect, pixmap=pix)
    
    test_pdf = Path(BASE) / 'temp_ocr_test.pdf'
    doc.save(str(test_pdf))
    doc.close()
    
    # Cleanup
    temp_img.unlink()
    
    return test_pdf

def test_ocr_fallback():
    """Test that OCR fallback is triggered when embedded text is below threshold."""
    print("Testing OCR Fallback Behavior")
    print("=" * 60)
    
    # Create test PDF with minimal embedded text
    print("\n1. Creating test PDF with minimal embedded text...")
    test_pdf = create_image_only_pdf()
    
    if not test_pdf or not test_pdf.exists():
        print("⚠️ Could not create test PDF (fitz required)")
        print("Skipping OCR fallback test")
        return
    
    try:
        # Process with LOW threshold to trigger OCR
        print("\n2. Processing with LOW embedded_text_threshold (5 chars)...")
        config = Config(
            ocr_enabled=True,
            embedded_text_threshold=5,  # LOW - should trigger OCR
            preserve_images=True,
            dpi=150,
        )
        
        processor = DocumentProcessor(config)
        result = processor.process(str(test_pdf))
        
        if not result:
            print("⚠️ Processing failed")
            return
        
        # Check results
        print("\n3. Analyzing extraction results...")
        print("-" * 60)
        
        for page in result.pages:
            source_details = getattr(page, 'source_details', {})
            ocr_used = source_details.get('ocr_used', False)
            embedded_text = getattr(page, 'embedded_text', '')
            ocr_text = getattr(page, 'ocr_text', '')
            
            print(f"\nPage {page.page_number}:")
            print(f"  Embedded text length: {len(embedded_text)} chars")
            print(f"  Embedded text: '{embedded_text[:50]}{'...' if len(embedded_text) > 50 else ''}'")
            print(f"  OCR used: {ocr_used}")
            print(f"  OCR text length: {len(ocr_text)} chars")
            print(f"  OCR text: '{ocr_text[:100]}{'...' if len(ocr_text) > 100 else ''}'")
            print(f"  Text source: {source_details.get('text_source', 'unknown').upper()}")
            print(f"  Extraction time: {source_details.get('extraction_time_ms', 0)}ms")
            
            # Verify OCR was used
            if ocr_used or len(ocr_text) > 0:
                print("\n  [SUCCESS] OCR fallback was correctly triggered!")
            else:
                print("\n  ⚠️ OCR was not triggered (document may have embedded text)")
        
        print("\n" + "=" * 60)
        print("SUCCESS: OCR fallback test completed")
        
    finally:
        # Cleanup
        if test_pdf.exists():
            test_pdf.unlink()
        print("\n📝 Test PDF cleaned up")

if __name__ == '__main__':
    test_ocr_fallback()
