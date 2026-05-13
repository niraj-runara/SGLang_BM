from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Dict, List, Optional, TypeVar

from openai import AsyncOpenAI


@dataclass
class StreamMetrics:
    ttft_ms: float
    total_s: float
    decode_s: float
    completion_tokens: int
    prompt_tokens: int
    ok: bool
    error: Optional[str] = None

    @property
    def decode_tps(self) -> float:
        if self.decode_s <= 0 or self.completion_tokens <= 0:
            return 0.0
        return self.completion_tokens / self.decode_s


def _instant_extra_body(use_instant: bool) -> Dict[str, Any]:
    if not use_instant:
        return {}
    # Model card: vLLM/SGLang instant mode
    return {"chat_template_kwargs": {"thinking": False}}


def _first_stream_signal(delta: Any) -> bool:
    if getattr(delta, "content", None):
        return True
    if getattr(delta, "reasoning_content", None):
        return True
    return False


async def chat_completion_stream_metrics(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    use_instant_mode: bool,
) -> StreamMetrics:
    t0 = time.perf_counter()
    ttft_ms = float("nan")
    first_token_recorded = False
    completion_tokens = 0
    prompt_tokens = 0
    err: Optional[str] = None
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            timeout=timeout_s,
            extra_body=_instant_extra_body(use_instant_mode),
        )
        t_first = None
        async for chunk in stream:
            if not first_token_recorded and chunk.choices:
                delta = chunk.choices[0].delta
                if _first_stream_signal(delta):
                    t_first = time.perf_counter()
                    ttft_ms = (t_first - t0) * 1000.0
                    first_token_recorded = True
            if chunk.usage:
                completion_tokens = int(chunk.usage.completion_tokens or 0)
                prompt_tokens = int(chunk.usage.prompt_tokens or 0)
        t_end = time.perf_counter()
        if t_first is None:
            t_first = t_end
        decode_s = max(t_end - t_first, 1e-9)
        total_s = t_end - t0
        if completion_tokens == 0:
            err = "usage_missing_or_zero_completion_tokens"
        return StreamMetrics(
            ttft_ms=ttft_ms,
            total_s=total_s,
            decode_s=decode_s,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            ok=completion_tokens > 0 and first_token_recorded,
            error=err,
        )
    except Exception as exc:  # pragma: no cover - network path
        t_end = time.perf_counter()
        return StreamMetrics(
            ttft_ms=float("nan"),
            total_s=t_end - t0,
            decode_s=0.0,
            completion_tokens=0,
            prompt_tokens=0,
            ok=False,
            error=repr(exc),
        )


T = TypeVar("T")


async def gather_limited(awaitables: List[Awaitable[T]], limit: int) -> List[T]:
    sem = asyncio.Semaphore(limit)

    async def _wrap(c: Awaitable[T]) -> T:
        async with sem:
            return await c

    return await asyncio.gather(*[_wrap(c) for c in awaitables])
