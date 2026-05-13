from __future__ import annotations

CODE_GEN_PROMPT = """You are a senior engineer. Implement a clean, well-documented solution.

Task:
- Language: Python 3.11+
- Implement a thread-safe in-memory rate limiter (token bucket).
- Include type hints, docstrings, and minimal unit tests in the same file under `if __name__ == "__main__":`.
- Do not use external packages beyond the standard library.

Output only the final source code file content.
"""

CHAT_TEMPLATES = [
    "Summarize the key risks of deploying MoE models on a single GPU for production chat.",
    "Explain how KV-cache memory grows with concurrent users in a chunked prefill server.",
    "List five practical checks before benchmarking LLM inference latency.",
    "Compare throughput-oriented vs latency-oriented batching for online inference.",
    "What metrics would you log to detect prefill-bound vs decode-bound regimes?",
]


def rotating_chat_prompt(i: int) -> str:
    return CHAT_TEMPLATES[i % len(CHAT_TEMPLATES)]


def build_token_prompt(tokenizer, num_tokens: int) -> str:
    """Build a single user message whose tokenized length is approximately num_tokens (no special tokens)."""
    seed = (
        "Benchmark filler paragraph. The quick brown fox jumps over the lazy dog. "
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Pack my box with five dozen liquor jugs. "
    )
    ids: list[int] = []
    chunk = tokenizer.encode(seed, add_special_tokens=False)
    if not chunk:
        chunk = tokenizer.encode("x", add_special_tokens=False)
    while len(ids) < num_tokens:
        need = num_tokens - len(ids)
        ids.extend(chunk[:need])
    ids = ids[:num_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)
