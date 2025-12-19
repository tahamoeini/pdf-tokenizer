import os
import json
import tempfile
from pathlib import Path

import fitz

from extract import process_pdf, OUTPUT_DIR


def _make_test_pdf(path: str):
    doc = fitz.open()
    # Page 1: long embedded text (no OCR)
    p = doc.new_page()
    p.insert_text(fitz.Point(72, 72), 'This is a long embedded text ' * 10, fontsize=12)
    # Page 2: image-only (forces render+OCR)
    p2 = doc.new_page()
    # create a small PNG and insert it
    from PIL import Image, ImageDraw
    img_path = os.path.join(os.path.dirname(path), 'tmp_img.png')
    im = Image.new('RGB', (400, 150), color=(255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([10, 10, 390, 140], outline='black', width=2)
    d.text((20, 20), 'Image Page', fill='black')
    im.save(img_path)
    r = fitz.Rect(50, 50, 300, 200)
    p2.insert_image(r, filename=img_path)
    # Page 3: short embedded text (should trigger OCR)
    p3 = doc.new_page()
    p3.insert_text(fitz.Point(72, 72), 'Short', fontsize=12)
    # Page 4: long embedded text
    p4 = doc.new_page()
    p4.insert_text(fitz.Point(72, 72), 'Another long page ' * 12, fontsize=12)
    # Page 5: image-only
    p5 = doc.new_page()
    img_path2 = os.path.join(os.path.dirname(path), 'tmp_img2.png')
    im2 = Image.new('RGB', (400, 150), color=(255, 255, 255))
    d2 = ImageDraw.Draw(im2)
    d2.rectangle([10, 10, 390, 140], outline='black', width=2)
    d2.text((20, 20), 'Image Page 2', fill='black')
    im2.save(img_path2)
    r2 = fitz.Rect(50, 50, 300, 200)
    p5.insert_image(r2, filename=img_path2)

    doc.save(path)
    doc.close()


def test_ordering_regression():
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "multi_test.pdf")
        _make_test_pdf(src)
        # Use parallelism to exercise ordering
        result = process_pdf(src, max_workers=4, enable_parallel=True, per_page_max_workers=2)
        assert result is not None
        # The processor writes output.json under processed_data/<stem>/output.json
        doc_stem = Path(result['filename']).stem
        out_dir = os.path.join(OUTPUT_DIR, doc_stem)
        json_path = os.path.join(out_dir, 'output.json')
        md_path = os.path.join(out_dir, 'output.md')
        assert os.path.exists(json_path), f"Missing {json_path}"
        with open(json_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        pages = payload.get('pages', [])
        page_nums = [p.get('page_number') for p in pages]
        assert page_nums == sorted(page_nums), f"Pages out of order in JSON: {page_nums}"

        # Check markdown has headings in ascending order
        assert os.path.exists(md_path), f"Missing {md_path}"
        with open(md_path, 'r', encoding='utf-8') as f:
            md = f.read()
        headings = []
        for line in md.splitlines():
            if line.strip().startswith('## Page'):
                try:
                    n = int(line.strip().split(' ')[2])
                    headings.append(n)
                except Exception:
                    continue
        assert headings == sorted(headings), f"Pages out of order in MD: {headings}"
