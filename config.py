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
    # Parallelism
    enable_parallel: bool = True
    max_workers: int = 0  # 0 = auto (based on CPU & memory)
    worker_mem_estimate_mb: int = 512  # rough memory estimate per worker
    # Per-page concurrency for embedded-image OCR (small, bounded to limit memory)
    per_page_max_workers: int = 2
    # Runtime metrics & soft-limiter
    log_runtime_metrics: bool = True
    cpu_soft_limit_percent: int = 80

    def wants(self, fmt: str) -> bool:
        return fmt.lower() in self.formats
