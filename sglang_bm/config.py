from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# Inference frameworks benchmark brief: 8x NVIDIA RTX PRO 6000 (Blackwell), 4-bit weights, Kimi-K2.6 on SGLang.
ASSIGNMENT_PLATFORM = "8x NVIDIA RTX PRO 6000 (Blackwell), tensor-parallel default 8"


@dataclass
class BenchDefaults:
    model: str = "moonshotai/Kimi-K2.6"
    base_url: str = "http://127.0.0.1:30000/v1"
    api_key: str = "sglang-bench"
    # Prefill sizes aligned with assignment windows (approximate tokenizer counts).
    context_targets: List[int] = field(default_factory=lambda: [1024, 8192, 32768, 65536, 131072])
    max_new_tokens_short: int = 128
    max_new_tokens_medium: int = 512
    max_new_tokens_code: int = 1024
    concurrency_levels: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    sustained_duration_s: float = 120.0
    sustained_concurrency: int = 8
    request_timeout_s: float = 7200.0
    gpu_sample_interval_s: float = 0.5
    # Primary GPU index for per-row nvidia-smi snapshots (TP rank 0 is a reasonable default on 8-GPU nodes).
    gpu_device_index: int = 0
    instant_mode: bool = True


DEFAULTS = BenchDefaults()
