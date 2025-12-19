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
            text = (p.get("text") or "").strip()
            if text:
                md.append(text + "\n\n")
            images = p.get("image_files") or []
            for rel in images:
                md.append(f"![Page {p['page_number']}]({rel})\n\n")
            md.append("---\n\n")

        out = os.path.join(self.base_dir, "output.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write("".join(md))
        return out
