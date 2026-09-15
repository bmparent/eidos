"""Bounded specialist fan-out with technical-failure-only retries."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


class SpecialistCallLimit(RuntimeError):
    pass


class BoundedSpecialistExecutor:
    def __init__(self, max_parallel: int, max_calls: int, max_retries: int) -> None:
        if max_parallel < 1 or max_calls < 1 or max_retries < 0:
            raise ValueError("invalid concurrency limits")
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.max_calls = max_calls
        self.max_retries = max_retries
        self.call_count = 0
        self._lock = asyncio.Lock()

    async def _one(self, name: str, operation: Callable[[], Awaitable[Any]]) -> tuple[str, Any]:
        async with self._lock:
            if self.call_count >= self.max_calls:
                raise SpecialistCallLimit("maximum specialist calls reached")
            self.call_count += 1
        async with self.semaphore:
            last: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    return name, await operation()
                except (TimeoutError, ConnectionError) as exc:
                    last = exc
                    if attempt == self.max_retries:
                        raise
            raise RuntimeError("unreachable") from last

    async def run(self, jobs: dict[str, Callable[[], Awaitable[Any]]]) -> dict[str, Any]:
        pairs = await asyncio.gather(*(self._one(name, operation) for name, operation in jobs.items()))
        return dict(pairs)
