import json
import os
from typing import List, Dict, Any
from dataclasses import asdict

from config import Config


class TextExporter:
    def __init__(self, base_dir: str):
        self.text_dir = os.path.join(base_dir, "text")
        os.makedirs(self.text_dir, exist_ok=True)
        self.written_files: List[str] = []

    def write_page(self, page_number: int, text: str) -> str:
        path = os.path.join(self.text_dir, f"page_{page_number:03}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text or "")
        self.written_files.append(path)
        return path

    def write_full(self, all_text: str) -> str:
        path = os.path.join(self.text_dir, "full_text.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(all_text or "")
        self.written_files.append(path)
        return path


class JsonExporter:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def write(self, payload: Dict[str, Any]) -> str:
        path = os.path.join(self.base_dir, "output.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path


class MarkdownExporter:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def write(self, pages: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        md: List[str] = []
        title = metadata.get("title") or metadata.get("filename") or "Document"
        md.append(f"# {title}\n\n")
        if metadata:
            author = metadata.get("author")
            if author and author != "Unknown":
                md.append(f"**Author:** {author}\n")
            page_count = metadata.get("page_count")
            if page_count:
                md.append(f"**Total Pages:** {page_count}\n")
            md.append("\n---\n\n")

        for p in pages:
            md.append(f"## Page {p['page_number']}\n\n")
            
            # Show source information
            source_details = p.get("source_details", {})
            if source_details:
                md.append(f"**Extraction Source:** {source_details.get('text_source', 'unknown').upper()}\n")
                if source_details.get('ocr_used'):
                    md.append(f"- OCR Fallback: Used (embedded text too short)\n")
                md.append(f"- Extraction Time: {source_details.get('extraction_time_ms', 0)}ms\n\n")
            
            # Embedded text section
            embedded_text = (p.get("embedded_text") or "").strip()
            if embedded_text:
                md.append(f"### 📄 Extracted Text (Embedded)\n\n{embedded_text}\n\n")
            
            # OCR text section
            ocr_text = (p.get("ocr_text") or "").strip()
            if ocr_text:
                md.append(f"### 🔍 Extracted Text (OCR)\n\n{ocr_text}\n\n")
            
            # Fallback to original text if no separation
            if not embedded_text and not ocr_text:
                text = (p.get("text") or "").strip()
                if text:
                    md.append(text + "\n\n")
            
            images = p.get("image_files") or []
            for rel in images:
                md.append(f"![Page {p['page_number']}]({rel})\n\n")

            # Embedded images metadata (with OCR text)
            embedded = p.get("embedded_images") or []
            if embedded:
                md.append(f"### 🖼️ Images in Page\n\n")
            for ei in embedded:
                rel = ei.get('path')
                text_e = ei.get('text') or ''
                if rel:
                    md.append(f"![Embedded image p{p['page_number']}]({rel})\n\n")
                if text_e:
                    md.append(f"**Text from image:**\n\n{text_e}\n\n")
            
            # Diagrams
            diagrams = p.get("diagrams") or []
            if diagrams:
                md.append(f"### 📊 Detected Diagrams\n\n")
            for d in diagrams:
                md.append(f"**Diagram:** {d.get('image')}\n\n")
                for shape in d.get('shapes', []):
                    bbox = shape.get('bbox')
                    if bbox:
                        md.append(f"- Shape bbox: {bbox}, type: {shape.get('type')}\n")
                        if shape.get('text'):
                            md.append(f"  - Text: {shape.get('text')}\n")
                if d.get('edges'):
                    md.append(f"- Edges (line segments): {len(d.get('edges'))}\n\n")
            md.append("---\n\n")

        out = os.path.join(self.base_dir, "output.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write("".join(md))
        return out
