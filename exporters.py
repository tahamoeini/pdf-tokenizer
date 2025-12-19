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
        # Reformat payload into sections -> pages -> content structure
        title = payload.get('metadata', {}).get('title') or payload.get('metadata', {}).get('input_filename') or 'Document'
        sections = []
        sec = {'name': title, 'pages': []}
        for p in payload.get('pages', []):
            page_num = p.get('page_number')
            # Build content: texts with image placeholders in order
            text = (p.get('text') or '').strip()
            embedded_text = (p.get('embedded_text') or '').strip()
            ocr_text = (p.get('ocr_text') or '').strip()

            # Primary text: prefer embedded_text if present, else ocr_text, else text
            primary_text = embedded_text or ocr_text or text or ''

            # Image placeholders — list of relative paths
            images = p.get('image_files') or []

            # For output simplicity, include a single combined_text field where images are represented as placeholders
            combined_text = primary_text
            for img in images:
                # keep placeholder token
                placeholder = f"[[IMAGE:{img}]]"
                if placeholder not in combined_text:
                    combined_text = combined_text + ("\n\n" if combined_text else "") + placeholder

            # Build images list with OCR text or failure note
            imgs_out = []
            for ei in p.get('embedded_images', []) + ([] if not p.get('embedded_images') else []):
                # embedded_images contains dicts with 'path' and 'text'
                path_rel = ei.get('path')
                text_e = (ei.get('text') or '').strip()
                if text_e:
                    imgs_out.append({'path': path_rel, 'ocr_text': text_e})
                else:
                    imgs_out.append({'path': path_rel, 'ocr_text': 'Unable to extract text from this image.'})

            # For any image files not in embedded_images, still report placeholder with failure notice
            known_paths = {ei.get('path') for ei in p.get('embedded_images', [])}
            for img in images:
                if img not in known_paths:
                    imgs_out.append({'path': img, 'ocr_text': 'Unable to extract text from this image.'})

            sec['pages'].append({
                'page_number': page_num,
                'text': combined_text,
                'images': imgs_out,
            })

        sections.append(sec)
        out_payload = {
            'sections': sections,
            'metadata': payload.get('metadata', {}),
        }

        path = os.path.join(self.base_dir, "output.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2, ensure_ascii=False)
        return path


class MarkdownExporter:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def write(self, pages: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        md_lines: List[str] = []
        title = metadata.get("title") or metadata.get("filename") or "Document"
        md_lines.append(f"# {title}\n\n")
        if metadata:
            author = metadata.get("author")
            if author and author != "Unknown":
                md_lines.append(f"**Author:** {author}\n")
            page_count = metadata.get("page_count")
            if page_count:
                md_lines.append(f"**Total Pages:** {page_count}\n")
            md_lines.append("\n---\n\n")

        # Single section (document-level)
        md_lines.append(f"## Section: {title}\n\n")

        for p in pages:
            page_num = p.get('page_number')
            md_lines.append(f"### Page {page_num}\n\n")

            # Primary combined text where images are placeholders
            combined_text = (p.get('text') or '').strip() or (p.get('embedded_text') or '').strip() or (p.get('ocr_text') or '').strip()
            images = p.get('image_files') or []
            # Append placeholders inline (do not attempt to replace unknown positions)
            if combined_text:
                md_lines.append(combined_text + "\n\n")
            for img in images:
                md_lines.append(f"[IMAGE: {img}]\n\n")

            # Now list images with OCR-extracted text (only text, no binaries)
            embedded = p.get('embedded_images') or []
            if images or embedded:
                md_lines.append("**Images:**\n\n")
            # Prefer embedded list entries (they include OCR text)
            seen = set()
            for ei in embedded:
                rel = ei.get('path')
                seen.add(rel)
                text_e = (ei.get('text') or '').strip()
                md_lines.append(f"- {rel}: {text_e if text_e else 'Unable to extract text from this image.'}\n")

            # Any image files that weren't in embedded_images
            for img in images:
                if img in seen:
                    continue
                md_lines.append(f"- {img}: Unable to extract text from this image.\n")

            md_lines.append("\n---\n\n")

        out = os.path.join(self.base_dir, "output.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write("".join(md_lines))
        return out
