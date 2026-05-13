from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional on non-GPU driver hosts
    pynvml = None


@dataclass
class GpuSample:
    ts: float
    util_gpu: Optional[float]
    mem_used_mib: Optional[float]
    mem_total_mib: Optional[float]


@dataclass
class GpuSampler:
    """Background NVML sampling for utilization and memory."""

    device_index: int = 0
    interval_s: float = 0.5
    _thread: Optional[threading.Thread] = None
    _stop: threading.Event = field(default_factory=threading.Event)
    samples: List[GpuSample] = field(default_factory=list)

    def start(self) -> None:
        if pynvml is None:
            return
        self._stop.clear()
        self.samples.clear()

        def _run() -> None:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            while not self._stop.is_set():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.samples.append(
                        GpuSample(
                            ts=time.time(),
                            util_gpu=float(util.gpu),
                            mem_used_mib=float(mem.used) / (1024**2),
                            mem_total_mib=float(mem.total) / (1024**2),
                        )
                    )
                except Exception:
                    self.samples.append(GpuSample(ts=time.time(), util_gpu=None, mem_used_mib=None, mem_total_mib=None))
                time.sleep(self.interval_s)
            pynvml.nvmlShutdown()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def summary(self) -> dict:
        if not self.samples:
            return {"available": False}
        utils = [s.util_gpu for s in self.samples if s.util_gpu is not None]
        used = [s.mem_used_mib for s in self.samples if s.mem_used_mib is not None]
        return {
            "available": True,
            "util_gpu_mean": float(sum(utils) / len(utils)) if utils else None,
            "util_gpu_max": float(max(utils)) if utils else None,
            "mem_used_mib_mean": float(sum(used) / len(used)) if used else None,
            "mem_used_mib_max": float(max(used)) if used else None,
            "mem_total_mib": self.samples[-1].mem_total_mib,
        }


def snapshot_all_gpus() -> List[Dict[str, Any]]:
    """One nvidia-smi query for every GPU (index, util %, used MiB, total MiB)."""
    ts = time.time()
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15.0,
        )
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx_s, util_s, used_s, total_s = parts
        try:
            rows.append(
                {
                    "ts": ts,
                    "index": int(float(idx_s)),
                    "utilization_gpu_pct": float(util_s),
                    "memory_used_mib": float(used_s),
                    "memory_total_mib": float(total_s),
                }
            )
        except ValueError:
            continue
    return rows


def snapshot_gpu(device_index: int = 0) -> GpuSample:
    """
    Prefer nvidia-smi so snapshots stay safe while a background NVML sampler thread runs.
    """
    ts = time.time()
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={device_index}",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
            ).strip()
            parts = [p.strip() for p in out.split(",")]
            if len(parts) == 3:
                util_s, used_s, total_s = parts
                return GpuSample(
                    ts=ts,
                    util_gpu=float(util_s),
                    mem_used_mib=float(used_s),
                    mem_total_mib=float(total_s),
                )
        except Exception:
            pass
    if pynvml is None:
        return GpuSample(ts=ts, util_gpu=None, mem_used_mib=None, mem_total_mib=None)
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return GpuSample(
            ts=ts,
            util_gpu=float(util.gpu),
            mem_used_mib=float(mem.used) / (1024**2),
            mem_total_mib=float(mem.total) / (1024**2),
        )
    finally:
        pynvml.nvmlShutdown()
