import os
import sys
import json
from pathlib import Path

# Adjust path
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import extract


def run():
    # Create sample PDF
    sample_pdf = BASE / 'resources' / 'sample.pdf'
    if not sample_pdf.exists():
        import create_sample_pdf  # noqa: F401
    assert sample_pdf.exists(), 'Sample PDF was not created'

    # Process single PDF
    res = extract.process_pdf(str(sample_pdf))
    assert res, 'Processing returned None'

    # Determine doc dir
    out_root = Path(extract.OUTPUT_DIR)
    doc_dir = out_root / 'sample'
    assert doc_dir.exists(), f'Document output directory missing: {doc_dir}'

    # Check text outputs
    text_dir = doc_dir / 'text'
    assert (text_dir / 'page_001.txt').exists(), 'Per-page text missing'
    assert (text_dir / 'full_text.txt').exists(), 'Merged text missing'

    # Check JSON and MD
    assert (doc_dir / 'output.json').exists(), 'output.json missing'
    assert (doc_dir / 'output.md').exists(), 'output.md missing'

    print('Smoke test passed. Outputs at:', str(doc_dir))


if __name__ == '__main__':
    run()
