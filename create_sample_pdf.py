import fitz
from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs('resources', exist_ok=True)

# Create a simple image (diagram)
img = Image.new('RGB', (600, 200), color=(255, 255, 255))
d = ImageDraw.Draw(img)
d.rectangle([50, 30, 550, 170], outline='black', width=2)
d.line([50, 100, 550, 100], fill='black', width=2)
d.text((60, 40), 'Sample Diagram: Flow A -> B', fill='black')
img_path = 'resources/sample_image.png'
img.save(img_path)

# Create a PDF with text, list, table and the image
doc = fitz.open()
page = doc.new_page()

# Heading
page.insert_text(fitz.Point(72, 72), 'SAMPLE DOCUMENT', fontsize=18, fontname='helv', fill=(0,0,0))

# Paragraph
text = 'This is a sample PDF created for testing the PDF Tokenizer. It contains headings, paragraphs, lists, a small table, and an embedded image which should be OCRed.'
page.insert_textbox(fitz.Rect(72, 100, 540, 200), text, fontsize=11)

# List
list_y = 220
for i, item in enumerate(['First item', 'Second item with more text', 'Third item']):
    page.insert_text(fitz.Point(90, list_y + i*16), f'- {item}', fontsize=11)

# Simple table (textual)
table_y = 280
page.insert_text(fitz.Point(90, table_y), 'Col1 | Col2 | Col3', fontsize=11)
page.insert_text(fitz.Point(90, table_y+14), '---- | ---- | ----', fontsize=11)
page.insert_text(fitz.Point(90, table_y+28), 'A1 | B1 | C1', fontsize=11)
page.insert_text(fitz.Point(90, table_y+42), 'A2 | B2 | C2', fontsize=11)

# Insert image
rect = fitz.Rect(72, 360, 420, 520)
page.insert_image(rect, filename=img_path)

out_path = 'resources/sample.pdf'
doc.save(out_path)
doc.close()
print('Created sample PDF at', out_path)
