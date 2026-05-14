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
| `summary_*.json` | Full roll-up: `meta`, all suite payloads under `suites`, optional `gpu` — see [§10](#10-result-files-reference). |

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

## 10. Result files reference

Each benchmark invocation writes JSON under `--output-dir` using one shared timestamp, e.g. `20260514-005404`, so files from the same run share the same suffix.

### 10.1 Output files (`full` + `--track-gpu` + `--snapshot-all-gpus`)

| Filename pattern | Written when | What it contains |
|------------------|----------------|------------------|
| `meta_<ts>.json` | Every run; **updated twice** if `--snapshot-all-gpus` | Run metadata: `ts`, `model`, `base_url`, `instant_mode`, `thinking_mode`, `assignment_platform`, `gpu_device_index`, `rubric_mapping`, `server_parallelism_note`. With `--snapshot-all-gpus`: `gpus_all_at_start` (before suites) and `gpus_all_at_end` (after suites, same file overwritten). |
| `latency_<ts>.json` | `latency` or `full` | One suite object: `suite` (`latency_sweep`), `rows[]` — per prefill target: TTFT, decode TPS, VRAM snapshots, `assignment_labels` per row. |
| `concurrency_mixed_<ts>.json` | `concurrency` or `full` (default modes include `mixed`) | `suite` (`concurrency_sweep`), `prefill_tokens`, `rows[]` for **mixed** mode (long-prefill parallel load). |
| `concurrency_chat_<ts>.json` | `concurrency` or `full` | Same shape; **chat** mode rows (`concurrent_chat_sessions` in rubric). |
| `concurrency_code_<ts>.json` | `concurrency` or `full` | Same shape; **code** mode rows (`parallel_code_generation` in rubric). |
| `sustained_<ts>.json` | `sustained` or `full` | Steady-load aggregate: duration, concurrency, completion count, error rate, decode TPS, TTFT stats, top-level `assignment_labels`. |
| `gpu_<ts>.json` | Only with `--track-gpu` | NVML time-series summary for `--gpu-device`: `series_summary` (`util_gpu_mean/max`, `mem_used_mib_mean/max`, etc.). |
| `summary_<ts>.json` | Every run | **Roll-up**: `meta` (final version after end snapshots), `suites[]` (latency + each concurrency + sustained objects), and `gpu` if `--track-gpu` was set. |

**Partial runs:** `latency` writes only `meta_*`, `latency_*`, `summary_*` (and `gpu_*` if tracked). `concurrency` omits latency/sustained files. `sustained` omits latency/concurrency files. Only `full` produces all seven suite-related files plus one `summary_*`.

### 10.2 Rubric scenario → where to look

Assignment-style scenarios map to **files** and to **`assignment_labels`** on rows (or on the sustained object).

| Benchmarking scenario (rubric) | Primary file | How to locate in JSON |
|--------------------------------|--------------|-------------------------|
| **Single-user inference** — short prompts up to 1k | `latency_<ts>.json` | Row with `assignment_labels` containing `short_prompt_up_to_1k` (default target 1024). |
| **Single-user inference** — medium prompts up to 8k | `latency_<ts>.json` | Row with `medium_prompt_up_to_8k` (default 8192). |
| **Single-user inference** — long / heavy prefill | `latency_<ts>.json` | Row with `long_context_prompt_heavy_prefill` (large prefill targets in the sweep). |
| **Long-context evaluation** — 32k class | `latency_<ts>.json` | Row with `long_context_evaluation_32k` (default prefill band includes 32768). |
| **Long-context evaluation** — 64k class | `latency_<ts>.json` | Row with `long_context_evaluation_64k` (default 65536). |
| **Long-context evaluation** — 128k class | `latency_<ts>.json` | Row with `long_context_evaluation_128k` (default 131072). |
| **Multi-user inference** — concurrent chat sessions | `concurrency_chat_<ts>.json` | Each `rows[]` entry includes `concurrent_chat_sessions` in `assignment_labels`. |
| **Multi-user inference** — parallel code generation | `concurrency_code_<ts>.json` | Rows include `parallel_code_generation`. |
| **Multi-user inference** — parallel long-prefill / mixed load | `concurrency_mixed_<ts>.json` | Rows include `parallel_requests_long_prefill`. |
| **Multi-user inference** — sustained throughput testing | `sustained_<ts>.json` | Top-level `assignment_labels` includes `sustained_throughput_testing`. |
| **System metrics** — all GPUs at start/end | `meta_<ts>.json` (and `summary_<ts>.json` → `meta`) | Arrays `gpus_all_at_start`, `gpus_all_at_end` (requires `--snapshot-all-gpus`). |
| **System metrics** — one GPU over time | `gpu_<ts>.json` | `series_summary` (requires `--track-gpu`). |

The same suite payloads appear inside `summary_<ts>.json` under `suites[]` (order: latency, then each concurrency mode in `--concurrency-modes` order, then sustained).

### 10.3 Top-level JSON shape per file

| File | Top-level keys (conceptually) |
|------|-------------------------------|
| `meta_<ts>.json` | Metadata fields listed in §10.1; optional `gpus_all_at_start` / `gpus_all_at_end`. |
| `latency_<ts>.json` | `suite`, `rows`. |
| `concurrency_*_<ts>.json` | `suite`, `prefill_tokens`, `rows`. |
| `sustained_<ts>.json` | `suite`, `assignment_labels`, duration/concurrency counters, aggregates, `decode_tps`, TTFT fields. |
| `gpu_<ts>.json` | `series_summary`. |
| `summary_<ts>.json` | `meta`, `suites` (array of suite objects), optional `gpu`. |

For a single place to archive or share a **full** run, use **`summary_<ts>.json`** plus the standalone copies of each suite file (they are duplicates of the objects inside `suites[]`, written for easy diffing and smaller per-topic files).
