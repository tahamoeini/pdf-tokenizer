import os
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class Config:
    # IO
    input_paths: List[str] = field(default_factory=list)
    output_dir: str = "processed_data"
    pipeline_mode: str = "full"  # full | fast-text

    # Processing
    ocr_enabled: bool = True
    dpi: int = 300
    lang: str = "eng"
    embedded_text_threshold: int = 40  # below this length, fallback to OCR
    preserve_images: bool = True
    enable_diagram_detection: bool = True
    enable_embedded_image_ocr: bool = True
    enable_per_page_text_exports: bool = True
    enable_nlp_enrichment: bool = True
    enable_combined_exports: bool = False

    # OCR quality heuristics
    ocr_text_min_length: int = 80
    ocr_alnum_ratio_min: float = 0.25

    # Outputs
    formats: Set[str] = field(default_factory=lambda: {"txt", "json", "md"})
    combined_jsonl_path: str = "combined_articles.jsonl"
    combined_txt_path: str = "combined_articles.txt"

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
    record_resource_metrics: bool = True

    def wants(self, fmt: str) -> bool:
        return fmt.lower() in self.formats

    def __post_init__(self):
        mode = (self.pipeline_mode or "full").strip().lower()
        if mode == "fast-text":
            # Force lightweight behavior unless the caller explicitly re-enabled
            self.enable_diagram_detection = False
            self.enable_embedded_image_ocr = False
            self.enable_per_page_text_exports = False
            self.enable_nlp_enrichment = False
            self.preserve_images = False
            # Fast mode defaults to combined outputs for downstream tooling
            self.enable_combined_exports = True
        else:
            self.pipeline_mode = "full"

        # Normalize combined output paths to live under output_dir if relative
        if self.combined_jsonl_path:
            self.combined_jsonl_path = self._resolve_output_path(self.combined_jsonl_path)
        if self.combined_txt_path:
            self.combined_txt_path = self._resolve_output_path(self.combined_txt_path)

    def _resolve_output_path(self, path: str) -> str:
        if not path:
            return path
        if os.path.isabs(path):
            return path
        return os.path.join(self.output_dir, path)
