"""
CLI driver for assignment-style metrics against a running SGLang OpenAI server.

Hardware scope (per team brief): 8x NVIDIA RTX PRO 6000 (Blackwell), 4-bit native quant, Kimi-K2.6.
The SGLang server should be launched with tensor parallel size 8 (see ``scripts/launch_sglang_kimi.sh``).
Per-row VRAM snapshots use ``--gpu-device`` (default 0); use ``--snapshot-all-gpus`` for every GPU via nvidia-smi.

Metrics (per request where applicable):
- TTFT (ms) from streaming first token (content or reasoning_content)
- Decode throughput (tokens/s) from usage.completion_tokens / decode wall time
- Aggregates: mean / p99, error rate
- GPU: mean/max utilization and VRAM (optional; snapshots use nvidia-smi)

Example:
  source .venv/bin/activate
  # Terminal A:
  bash scripts/launch_sglang_kimi.sh
  # Terminal B:
  python -m sglang_bm.run_benchmark --output-dir results/run1 full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from sglang_bm.bench_client import chat_completion_stream_metrics, gather_limited
from sglang_bm.config import ASSIGNMENT_PLATFORM, DEFAULTS
from sglang_bm.gpu_metrics import GpuSampler, snapshot_all_gpus, snapshot_gpu
from sglang_bm.prompts import CODE_GEN_PROMPT, build_token_prompt, rotating_chat_prompt


def _assignment_labels_latency(prefill_target: int) -> List[str]:
    """Map a prefill token target to the assignment table rows (Benchmarking Scenarios)."""
    labels: List[str] = ["single_user_inference"]
    if prefill_target <= 1024:
        labels.append("short_prompt_up_to_1k")
    elif prefill_target <= 8192:
        labels.append("medium_prompt_up_to_8k")
    else:
        labels.append("long_context_prompt_heavy_prefill")
    # Long-context evaluation row (32k / 64k / 128k)
    if 28_000 <= prefill_target <= 40_000:
        labels.append("long_context_evaluation_32k")
    elif 56_000 <= prefill_target <= 72_000:
        labels.append("long_context_evaluation_64k")
    elif prefill_target >= 120_000:
        labels.append("long_context_evaluation_128k")
    return labels


def _assignment_labels_concurrency(mode: str) -> List[str]:
    labels: List[str] = ["multi_user_inference"]
    if mode == "chat":
        labels.append("concurrent_chat_sessions")
    elif mode == "code":
        labels.append("parallel_code_generation")
    else:
        labels.append("parallel_requests_long_prefill")
    return labels


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def _summarize_tps(samples: List[float]) -> Dict[str, float]:
    clean = [x for x in samples if x > 0 and not np.isnan(x)]
    if not clean:
        return {"mean": float("nan"), "p99": float("nan")}
    return {"mean": float(np.mean(clean)), "p99": _percentile(clean, 99.0)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


async def run_latency_suite(
    client: AsyncOpenAI,
    tokenizer,
    *,
    model: str,
    context_targets: List[int],
    repeats: int,
    max_new_tokens: int,
    temperature: float,
    timeout_s: float,
    use_instant: bool,
    gpu_device: int,
) -> Dict[str, Any]:
    rows = []
    idle = snapshot_gpu(gpu_device)
    for n_ctx in tqdm(context_targets, desc="context sweep"):
        try:
            prompt = build_token_prompt(tokenizer, n_ctx)
        except Exception as exc:
            rows.append(
                {
                    "scenario": "latency_sweep",
                    "assignment_labels": _assignment_labels_latency(n_ctx),
                    "prefill_tokens_target": n_ctx,
                    "ok": False,
                    "error": f"prompt_build_failed:{exc!r}",
                }
            )
            continue
        actual = len(tokenizer.encode(prompt, add_special_tokens=False))
        ttfts: List[float] = []
        tpss: List[float] = []
        oks = 0
        for _ in range(repeats):
            m = await chat_completion_stream_metrics(
                client,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
                use_instant_mode=use_instant,
            )
            if m.ok and not np.isnan(m.ttft_ms):
                ttfts.append(m.ttft_ms)
                tpss.append(m.decode_tps)
                oks += 1
            await asyncio.sleep(0.05)
        under_load = snapshot_gpu(gpu_device)
        rows.append(
            {
                "scenario": "latency_sweep",
                "assignment_labels": _assignment_labels_latency(n_ctx),
                "prefill_tokens_target": n_ctx,
                "prefill_tokens_actual": actual,
                "repeats": repeats,
                "ok_rate": oks / max(repeats, 1),
                "ttft_ms_mean": float(np.mean(ttfts)) if ttfts else float("nan"),
                "ttft_ms_p99": _percentile(ttfts, 99.0) if ttfts else float("nan"),
                "decode_tps": _summarize_tps(tpss),
                "max_new_tokens": max_new_tokens,
                "vram_mib_idle": idle.mem_used_mib,
                "vram_mib_after": under_load.mem_used_mib,
            }
        )
    return {"suite": "latency_sweep", "rows": rows}


async def run_concurrency_suite(
    client: AsyncOpenAI,
    tokenizer,
    *,
    model: str,
    prefill_tokens: int,
    concurrency_levels: List[int],
    max_new_tokens: int,
    temperature: float,
    timeout_s: float,
    use_instant: bool,
    gpu_device: int,
    mode: str,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    if mode == "chat":
        base_prompt = build_token_prompt(tokenizer, min(prefill_tokens, 8192))
    elif mode == "code":
        base_prompt = CODE_GEN_PROMPT
    else:
        base_prompt = build_token_prompt(tokenizer, prefill_tokens)

    for conc in tqdm(concurrency_levels, desc=f"concurrency ({mode})"):
        idle = snapshot_gpu(gpu_device)

        async def one(idx: int):
            if mode == "chat":
                msg = rotating_chat_prompt(idx) + "\n\n" + base_prompt[:2000]
            elif mode == "code":
                msg = base_prompt
            else:
                msg = build_token_prompt(tokenizer, prefill_tokens)
            return await chat_completion_stream_metrics(
                client,
                model=model,
                messages=[{"role": "user", "content": msg}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
                use_instant_mode=use_instant,
            )

        awaitables = [one(i) for i in range(conc)]
        t0 = time.perf_counter()
        results = await gather_limited(awaitables, conc)
        wall = time.perf_counter() - t0
        load = snapshot_gpu(gpu_device)

        ttfts = [m.ttft_ms for m in results if m.ok and not np.isnan(m.ttft_ms)]
        tpss = [m.decode_tps for m in results if m.ok]
        toks = sum(m.completion_tokens for m in results if m.ok)
        oks = sum(1 for m in results if m.ok)
        errs = [m.error for m in results if not m.ok]

        rows.append(
            {
                "scenario": "concurrency_sweep",
                "assignment_labels": _assignment_labels_concurrency(mode),
                "mode": mode,
                "concurrency": conc,
                "wall_s": wall,
                "ok": oks,
                "errors_sample": errs[:3],
                "total_completion_tokens": toks,
                "system_tps_mean": toks / wall if wall > 0 else 0.0,
                "ttft_ms_mean": float(np.mean(ttfts)) if ttfts else float("nan"),
                "ttft_ms_p99": _percentile(ttfts, 99.0) if ttfts else float("nan"),
                "decode_tps_per_req": _summarize_tps(tpss),
                "vram_mib_idle": idle.mem_used_mib,
                "vram_mib_peak_sample": load.mem_used_mib,
            }
        )
        await asyncio.sleep(0.2)

    return {"suite": "concurrency_sweep", "prefill_tokens": prefill_tokens, "rows": rows}


async def run_sustained(
    client: AsyncOpenAI,
    *,
    model: str,
    duration_s: float,
    concurrency: int,
    max_new_tokens: int,
    temperature: float,
    timeout_s: float,
    use_instant: bool,
) -> Dict[str, Any]:
    t_end = time.perf_counter() + duration_s
    lock = asyncio.Lock()
    completed = 0
    errors = 0
    tpss: List[float] = []
    ttfts: List[float] = []

    async def worker(wid: int):
        nonlocal completed, errors
        i = 0
        while time.perf_counter() < t_end:
            msg = rotating_chat_prompt(wid + i)
            m = await chat_completion_stream_metrics(
                client,
                model=model,
                messages=[{"role": "user", "content": msg}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
                use_instant_mode=use_instant,
            )
            async with lock:
                if m.ok:
                    completed += 1
                    tpss.append(m.decode_tps)
                    if not np.isnan(m.ttft_ms):
                        ttfts.append(m.ttft_ms)
                else:
                    errors += 1
            i += 1

    await asyncio.gather(*[asyncio.create_task(worker(c)) for c in range(concurrency)])
    total = completed + errors
    return {
        "suite": "sustained",
        "assignment_labels": ["multi_user_inference", "sustained_throughput_testing"],
        "duration_s": duration_s,
        "concurrency": concurrency,
        "completed": completed,
        "errors": errors,
        "error_rate": errors / total if total else 0.0,
        "decode_tps": _summarize_tps(tpss),
        "ttft_ms_mean": float(np.mean(ttfts)) if ttfts else float("nan"),
        "ttft_ms_p99": _percentile(ttfts, 99.0) if ttfts else float("nan"),
    }


async def _warmup(client: AsyncOpenAI, *, model: str, timeout_s: float, use_instant: bool) -> None:
    await chat_completion_stream_metrics(
        client,
        model=model,
        messages=[{"role": "user", "content": "Say the word 'ready' only."}],
        max_tokens=8,
        temperature=0.0,
        timeout_s=timeout_s,
        use_instant_mode=use_instant,
    )


async def async_main(args: argparse.Namespace) -> None:
    model = args.model
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    use_instant = not args.thinking_mode
    await _warmup(client, model=model, timeout_s=args.timeout, use_instant=use_instant)

    gpu_sampler = GpuSampler(device_index=args.gpu_device, interval_s=args.gpu_interval)
    if args.track_gpu:
        gpu_sampler.start()

    out_dir = Path(args.output_dir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    meta = {
        "ts": ts,
        "assignment_platform": ASSIGNMENT_PLATFORM,
        "gpu_device_index": args.gpu_device,
        "model": model,
        "base_url": args.base_url,
        "instant_mode": use_instant,
        "thinking_mode": args.thinking_mode,
        "framework": "SGLang",
        "rubric_mapping": {
            "single_user_inference": "latency suite; rows tagged short_prompt_up_to_1k | medium_prompt_up_to_8k | long_context_evaluation_*",
            "multi_user_inference": "concurrency suite (chat|code|mixed) + sustained suite",
            "gap": "Concurrent chat is single-turn per request, not multi-turn session history growth",
        },
        "server_parallelism_note": "Default launch uses TP_SIZE=8 across eight GPUs (one sharded replica).",
    }
    if args.snapshot_all_gpus:
        meta["gpus_all_at_start"] = snapshot_all_gpus()
    _write_json(out_dir / f"meta_{ts}.json", meta)

    results: Dict[str, Any] = {"meta": meta, "suites": []}

    if args.command in ("latency", "full"):
        suite = await run_latency_suite(
            client,
            tokenizer,
            model=model,
            context_targets=args.context_targets,
            repeats=args.repeats,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout,
            use_instant=use_instant,
            gpu_device=args.gpu_device,
        )
        results["suites"].append(suite)
        _write_json(out_dir / f"latency_{ts}.json", suite)

    if args.command in ("concurrency", "full"):
        for mode in args.concurrency_modes:
            suite = await run_concurrency_suite(
                client,
                tokenizer,
                model=model,
                prefill_tokens=args.concurrency_prefill,
                concurrency_levels=args.concurrency_levels,
                max_new_tokens=args.concurrency_max_tokens,
                temperature=args.temperature,
                timeout_s=args.timeout,
                use_instant=use_instant,
                gpu_device=args.gpu_device,
                mode=mode,
            )
            results["suites"].append(suite)
            _write_json(out_dir / f"concurrency_{mode}_{ts}.json", suite)

    if args.command in ("sustained", "full"):
        suite = await run_sustained(
            client,
            model=model,
            duration_s=args.sustained_duration,
            concurrency=args.sustained_concurrency,
            max_new_tokens=args.sustained_max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout,
            use_instant=use_instant,
        )
        results["suites"].append(suite)
        _write_json(out_dir / f"sustained_{ts}.json", suite)

    if args.track_gpu:
        gpu_sampler.stop()
        results["gpu"] = {"series_summary": gpu_sampler.summary()}
        _write_json(out_dir / f"gpu_{ts}.json", results["gpu"])

    if args.snapshot_all_gpus:
        meta["gpus_all_at_end"] = snapshot_all_gpus()
        results["meta"] = meta
        _write_json(out_dir / f"meta_{ts}.json", meta)

    _write_json(out_dir / f"summary_{ts}.json", results)
    print(json.dumps({"wrote": str(out_dir), "meta": meta}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SGLang / Kimi-K2.6 inference benchmark harness")
    p.add_argument("--base-url", default=DEFAULTS.base_url)
    p.add_argument("--api-key", default=DEFAULTS.api_key)
    p.add_argument("--model", default=DEFAULTS.model)
    p.add_argument(
        "--tokenizer-path",
        default=DEFAULTS.model,
        help="HF id or local path used only to size prompts (download/cache).",
    )
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout", type=float, default=DEFAULTS.request_timeout_s)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--repeats", type=int, default=3, help="Repeats per context in latency sweep.")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULTS.max_new_tokens_short)
    p.add_argument(
        "--context-targets",
        type=int,
        nargs="+",
        default=DEFAULTS.context_targets,
        help="Approximate prefill token counts for long-context evaluation.",
    )
    p.add_argument(
        "--concurrency-levels",
        type=int,
        nargs="+",
        default=DEFAULTS.concurrency_levels,
    )
    p.add_argument("--concurrency-prefill", type=int, default=4096)
    p.add_argument("--concurrency-max-tokens", type=int, default=256)
    p.add_argument(
        "--concurrency-modes",
        nargs="+",
        default=["mixed", "chat", "code"],
        choices=["mixed", "chat", "code"],
        help="mixed=long prefill; chat=multi-user short-ish; code=parallel codegen style.",
    )
    p.add_argument("--sustained-duration", type=float, default=DEFAULTS.sustained_duration_s)
    p.add_argument("--sustained-concurrency", type=int, default=DEFAULTS.sustained_concurrency)
    p.add_argument("--sustained-max-tokens", type=int, default=128)
    p.add_argument("--thinking-mode", action="store_true", help="Use model thinking (no instant override).")
    p.add_argument("--track-gpu", action="store_true", help="Background NVML sampling (Linux/NVIDIA).")
    p.add_argument(
        "--snapshot-all-gpus",
        action="store_true",
        help="nvidia-smi snapshot for every GPU at start/end of the run (capacity on 8-GPU nodes).",
    )
    p.add_argument("--gpu-device", type=int, default=DEFAULTS.gpu_device_index)
    p.add_argument("--gpu-interval", type=float, default=DEFAULTS.gpu_sample_interval_s)

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("latency", help="TTFT / decode TPS vs prefill length")
    sub.add_parser("concurrency", help="Throughput degradation vs concurrency")
    sub.add_parser("sustained", help="Runtime stability / error rate under steady load")
    sub.add_parser("full", help="Run latency + concurrency + sustained")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if os.environ.get("HF_TOKEN"):
        # Tokenizer may hit gated configs; OpenAI server already loaded weights.
        pass
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
