# SGLang benchmark harness (Kimi-K2.6)

OpenAI-compatible client benchmarks against a **running SGLang** server for [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6). Intended for the inference-frameworks assignment (**8× NVIDIA RTX PRO 6000 Blackwell**, tensor-parallel serving, 4-bit weights, long context).

Run the steps **in order** on the GPU machine (or use two shells after setup).

---

## 0. Prerequisites

- Linux with **NVIDIA driver** working (`nvidia-smi` prints GPU name and memory).
- **Python 3.10–3.12**.
- Enough disk for model cache (Hugging Face default: `~/.cache/huggingface`).
- If the Hub requires it: `export HF_TOKEN=...`

---

## 1. Clone / enter the repo
git clone https://github.com/niraj-runara/SGLang_BM.git
```bash
cd /path/to/sglangBM
```

Make scripts executable once:

```bash
chmod +x scripts/setup_gpu_host.sh scripts/download_kimi.sh scripts/launch_sglang_kimi.sh
```

---

## 2. Install dependencies (one-time)

This creates `.venv`, installs CUDA **PyTorch** (wheel index overridable), **SGLang**, and benchmark packages.

```bash
bash scripts/setup_gpu_host.sh
```

Activate the environment **for every new shell**:

```bash
source .venv/bin/activate
```

**Optional overrides**

| Variable | Purpose |
|----------|---------|
| `VENV` | Alternate venv path (default: `./.venv`). |
| `TORCH_INDEX_URL` | PyTorch wheel index if `cu128` does not match your image (example: `https://download.pytorch.org/whl/cu126`). |
| `SKIP_TORCH=1` | Skip PyTorch install if the image already has a matching `torch`. |
| `SKIP_SGLANG=1` | Skip SGLang install. |

---

## 3. Download the model weights

With the venv active, download Kimi-K2.6 to a local model directory:

```bash
source .venv/bin/activate
export HF_TOKEN=...   # if needed for model download
bash scripts/download_kimi.sh
```

By default, weights are written to `models/Kimi-K2.6`, and the serving script reads from that path. To use a different disk or mount, set `MODEL_DIR` during download and `MODEL_PATH` during launch:

```bash
MODEL_DIR=/workspace/models/Kimi-K2.6 bash scripts/download_kimi.sh
MODEL_PATH=/workspace/models/Kimi-K2.6 bash scripts/launch_sglang_kimi.sh
```

---

## 4. Start the SGLang server (keep this running)

Use a **dedicated terminal** (Terminal A). With the venv active:

```bash
source .venv/bin/activate
bash scripts/launch_sglang_kimi.sh
```

Default API base: `http://127.0.0.1:30000/v1` (model id `moonshotai/Kimi-K2.6`).

**8-GPU tensor parallel (default):** `scripts/launch_sglang_kimi.sh` sets **`TP_SIZE=8`** so one SGLang process shards **one** Kimi replica across all eight RTX PRO 6000 GPUs. Ensure all eight are visible (`nvidia-smi` lists eight devices). For multi-node or different sharding, follow your infra’s SGLang layout and adjust `TP_SIZE` / launch flags accordingly.

**Smoke test on a single GPU** (not the assignment topology):

```bash
TP_SIZE=1 bash scripts/launch_sglang_kimi.sh
```

**If you hit OOM at long context**, tune env vars before launch, for example:

```bash
CONTEXT_LENGTH=65536 MEM_FRACTION_STATIC=0.82 bash scripts/launch_sglang_kimi.sh
```

Optional: restrict visible devices (example: first eight indices):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash scripts/launch_sglang_kimi.sh
```

Wait until the server has finished loading and is accepting HTTP traffic.

---

## 5. Run benchmarks (second terminal)

Open **Terminal B**, activate the same venv:

```bash
cd /path/to/sglangBM
source .venv/bin/activate
export HF_TOKEN=...   # only if the tokenizer download needs it
```

### 5a. Full suite (recommended first run)

Runs **latency** + **concurrency** + **sustained** workloads:

```bash
python -m sglang_bm.run_benchmark --output-dir results/run1 --track-gpu --snapshot-all-gpus full
```

`--snapshot-all-gpus` records **every** GPU’s VRAM/util via `nvidia-smi` at the start and end of the run (useful for 8-GPU capacity tables). `--track-gpu` samples **one** GPU index (default `0`) over time via NVML.

### 5b. Run suites individually (same order as the table)

1. **Single-user + long-context (TTFT / decode TPS vs prefill length)**

   ```bash
   python -m sglang_bm.run_benchmark --output-dir results/latency1 latency
   ```

2. **Multi-user load (parallel requests)**

   ```bash
   python -m sglang_bm.run_benchmark --output-dir results/conc1 concurrency
   ```

3. **Sustained throughput / stability**

   ```bash
   python -m sglang_bm.run_benchmark --output-dir results/sustained1 sustained
   ```

**Note:** Put global flags (`--output-dir`, `--track-gpu`, `--snapshot-all-gpus`, …) *before* the subcommand (`full`, `latency`, …).

---

## 6. Where results go

Under `--output-dir` (default `results/`), each run writes timestamped JSON, for example:

| File pattern | Contents |
|--------------|----------|
| `meta_*.json` | Model, URL, instant vs thinking, assignment platform, rubric notes; with `--snapshot-all-gpus`, includes `gpus_all_at_start` / `gpus_all_at_end`. |
| `latency_*.json` | Single-user prefill sweep, `assignment_labels` per row. |
| `concurrency_<mode>_*.json` | Load sweep per mode (`mixed`, `chat`, `code`). |
| `sustained_*.json` | Steady-load error rate and TPS aggregates. |
| `gpu_*.json` | Optional NVML time series summary (`--track-gpu`). |
| `summary_*.json` | Full roll-up: `meta`, all suite payloads under `suites`, optional `gpu` — see [§10](#10-sample-benchmark-results) for a tabulated sample run. |

---

## 7. Common CLI flags

| Flag | Meaning |
|------|---------|
| `--base-url` | If the server is not on `http://127.0.0.1:30000/v1`. |
| `--thinking-mode` | Disables “instant” `chat_template_kwargs` override (default is instant mode for cross-framework parity). |
| `--context-targets` | Space-separated prefill token targets for `latency` (default includes 1k, 8k, 32k, 64k, 128k-class). |
| `--concurrency-levels` | Parallelism schedule for `concurrency`. |
| `--concurrency-modes` | Subset of `mixed` `chat` `code`. |
| `--sustained-duration` | Seconds for `sustained`. |
| `--track-gpu` | Background sampling on `--gpu-device` (NVML side thread). |
| `--snapshot-all-gpus` | Start/end `nvidia-smi` snapshot for **all** GPUs (8× capacity reporting). |
| `--gpu-device` | GPU index for per-row VRAM fields in latency/concurrency (default `0`). |

Full help:

```bash
python -m sglang_bm.run_benchmark --help
python -m sglang_bm.run_benchmark full --help
```

---

## 8. What is *not* simulated by default

- **Multi-turn chat sessions** (growing `messages` history per user) are not the same as **many parallel single-turn** chats. The `concurrency` `chat` mode models concurrent *single-turn* users. Extend the harness if graders require explicit multi-turn KV growth.

---

## 9. Quick troubleshooting

| Symptom | What to try |
|---------|-------------|
| `nvidia-smi` missing | Install the proprietary NVIDIA driver on the host. |
| Server OOM on long context | Reduce `CONTEXT_LENGTH`, lower `MEM_FRACTION_STATIC`, or increase `TP_SIZE`. |
| Tokenizer / model download fails | `export HF_TOKEN=...`, check disk and network. |
| Benchmark cannot connect | Confirm server URL/port and firewall; match `--base-url`. |
| `sglang[all]` install fails | Script falls back to `pip install sglang`; install extras per SGLang docs. |

---

## 10. Sample benchmark results

Values below are copied from one full harness run (`results/run1/`, timestamp **`20260514-005404`**). Re-run `python -m sglang_bm.run_benchmark … full` and refresh this section from the new JSON if you want the README to stay in sync with latest hardware.

**Run context (from `meta_20260514-005404.json`):** `moonshotai/Kimi-K2.6`, `http://127.0.0.1:30000/v1`, instant mode (thinking off), `8x NVIDIA RTX PRO 6000 (Blackwell), tensor-parallel default 8`, tracked GPU index **0**.

### Single-user latency (`latency_20260514-005404.json`)

| Rubric (labels) | Prefill target | Prefill actual | OK rate | TTFT mean (ms) | TTFT p99 (ms) | Decode TPS mean | Decode TPS p99 | VRAM after (MiB, GPU 0) |
|-----------------|----------------:|---------------:|--------:|---------------:|--------------:|------------------:|---------------:|------------------------:|
| short ≤1k | 1024 | 1002 | 1.00 | 304.46 | 612.42 | 64.79 | 64.95 | 89033 |
| medium ≤8k | 8192 | 8010 | 1.00 | 1494.00 | 3939.00 | 41.95 | 42.03 | 90381 |
| long prefill + **32k eval** | 32768 | 32040 | 1.00 | 6844.01 | 18771.90 | 18.17 | 18.18 | 91281 |
| long prefill + **64k eval** | 65536 | 64080 | 1.00 | 13701.95 | 37449.66 | 10.05 | 10.06 | 91283 |
| long prefill + **128k eval** | 131072 | 128160 | 1.00 | 45658.76 | 128616.56 | 5.42 | 5.43 | 91287 |

### Concurrency — mixed, prefill 4096 (`concurrency_mixed_20260514-005404.json`)

| Concurrency | OK reqs | Wall (s) | System TPS mean | TTFT mean (ms) | TTFT p99 (ms) | Decode TPS / req mean | Decode TPS / req p99 |
|------------:|--------:|---------:|----------------:|---------------:|--------------:|----------------------:|---------------------:|
| 1 | 1 | 3.32 | 45.14 | 162.34 | 162.34 | 52.41 | 52.41 |
| 2 | 2 | 5.48 | 64.92 | 537.07 | 550.54 | 46.41 | 49.21 |
| 4 | 4 | 6.32 | 108.23 | 668.89 | 1070.59 | 43.20 | 43.94 |
| 8 | 8 | 9.46 | 148.99 | 1349.09 | 2225.57 | 33.63 | 36.06 |
| 16 | 16 | 13.89 | 189.23 | 2659.45 | 4522.95 | 23.03 | 26.01 |
| 32 | 32 | 24.13 | 232.23 | 5382.73 | 9605.31 | 15.27 | 17.86 |

### Concurrency — chat (`concurrency_chat_20260514-005404.json`)

| Concurrency | OK reqs | Wall (s) | System TPS mean | TTFT mean (ms) | TTFT p99 (ms) | Decode TPS / req mean | Decode TPS / req p99 |
|------------:|--------:|---------:|----------------:|---------------:|--------------:|----------------------:|---------------------:|
| 1 | 1 | 4.19 | 61.05 | 373.49 | 373.49 | 67.03 | 67.03 |
| 2 | 2 | 4.68 | 109.31 | 400.11 | 402.57 | 59.78 | 59.80 |
| 4 | 4 | 5.76 | 177.46 | 653.94 | 731.16 | 50.20 | 50.93 |
| 8 | 8 | 7.27 | 276.59 | 454.86 | 480.94 | 37.57 | 37.73 |
| 16 | 16 | 10.35 | 387.72 | 276.84 | 290.00 | 25.48 | 25.87 |
| 32 | 32 | 16.35 | 493.08 | 284.73 | 297.33 | 15.95 | 16.03 |

### Concurrency — code (`concurrency_code_20260514-005404.json`)

| Concurrency | OK reqs | Wall (s) | System TPS mean | TTFT mean (ms) | TTFT p99 (ms) | Decode TPS / req mean | Decode TPS / req p99 |
|------------:|--------:|---------:|----------------:|---------------:|--------------:|----------------------:|---------------------:|
| 1 | 1 | 3.76 | 68.11 | 114.31 | 114.31 | 70.26 | 70.26 |
| 2 | 2 | 4.20 | 122.05 | 179.14 | 182.75 | 63.79 | 63.82 |
| 4 | 4 | 4.82 | 212.30 | 174.02 | 177.60 | 55.11 | 55.16 |
| 8 | 8 | 6.45 | 317.58 | 186.32 | 191.69 | 40.93 | 40.96 |
| 16 | 16 | 9.49 | 431.74 | 193.54 | 201.92 | 27.59 | 27.61 |
| 32 | 32 | 15.52 | 527.86 | 302.71 | 314.74 | 16.86 | 16.86 |

### Sustained (`sustained_20260514-005404.json`)

| Duration (s) | Concurrency | Completed | Errors | Error rate | Decode TPS mean | Decode TPS p99 | TTFT mean (ms) | TTFT p99 (ms) |
|---------------:|--------------:|----------:|-------:|-----------:|----------------:|---------------:|---------------:|--------------:|
| 120 | 8 | 280 | 0 | 0.00 | 38.99 | 39.95 | 185.53 | 295.10 |

### GPU time series — device 0 (`gpu_20260514-005404.json`)

| Metric | Value |
|--------|------:|
| Util mean (%) | 91.60 |
| Util max (%) | 100.0 |
| Mem used mean (MiB) | 91870.2 |
| Mem used max (MiB) | 91925.3 |
| Mem total (MiB) | 97887 |

### All GPUs — VRAM / util snapshot (`meta_20260514-005404.json`)

**At start of run**

| GPU | Util (%) | Mem used (MiB) | Mem total (MiB) |
|----:|---------:|---------------:|----------------:|
| 0 | 63 | 88921 | 97887 |
| 1 | 96 | 88269 | 97887 |
| 2 | 97 | 88269 | 97887 |
| 3 | 68 | 88269 | 97887 |
| 4 | 57 | 88269 | 97887 |
| 5 | 97 | 88269 | 97887 |
| 6 | 95 | 88269 | 97887 |
| 7 | 95 | 88269 | 97887 |

**At end of run**

| GPU | Util (%) | Mem used (MiB) | Mem total (MiB) |
|----:|---------:|---------------:|----------------:|
| 0 | 0 | 91289 | 97887 |
| 1 | 0 | 90637 | 97887 |
| 2 | 0 | 90637 | 97887 |
| 3 | 0 | 90637 | 97887 |
| 4 | 0 | 90637 | 97887 |
| 5 | 0 | 90637 | 97887 |
| 6 | 0 | 90637 | 97887 |
| 7 | 0 | 90637 | 97887 |

Raw JSON for this run: `meta_*`, `latency_*`, `concurrency_*`, `sustained_*`, `gpu_*`, `summary_*` under `results/run1/`. File naming is summarized in [§6](#6-where-results-go).
