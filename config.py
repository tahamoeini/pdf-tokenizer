from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class Config:
    # IO
    input_paths: List[str] = field(default_factory=list)
    output_dir: str = "processed_data"

    # Processing
    ocr_enabled: bool = True
    dpi: int = 300
    lang: str = "eng"
    embedded_text_threshold: int = 40  # below this length, fallback to OCR
    preserve_images: bool = True

    # Outputs
    formats: Set[str] = field(default_factory=lambda: {"txt", "json", "md"})

    # Misc
    verbose: bool = False
    self_check: bool = False

    def wants(self, fmt: str) -> bool:
        return fmt.lower() in self.formats
