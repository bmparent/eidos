"""Stream generators for the Eidos RNG null-proof harness.

Generators expose only sequential values to the runner.  The runner records
metadata separately so the proof loop can enforce predict-then-reveal without
passing future values, seeds, or generator internals into the predictor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import random
import secrets
from typing import Callable, Iterator, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency fallback
    np = None


@dataclass(frozen=True)
class StreamSpec:
    name: str
    target_space: str
    size: int
    reproducible: bool
    seed: Optional[int]
    algorithm: str
    category: str
    factory: Callable[[], Iterator[int]]


def _space_size(target_space: str) -> int:
    return {"bits": 2, "digits": 10, "integers": 100, "bytes": 256}[target_space]


def _rng(seed: Optional[int], salt: str) -> random.Random:
    return random.Random(f"{seed}:{salt}")


def repeating_sequence(seed: int | None = None, target_space: str = "digits") -> StreamSpec:
    size = _space_size(target_space)
    def gen() -> Iterator[int]:
        i = 0
        while True:
            yield i % size
            i += 1
    return StreamSpec("repeating_sequence", target_space, size, True, seed, "i mod K", "structured", gen)


def sine_quantized(seed: int | None = None, target_space: str = "digits") -> StreamSpec:
    size = _space_size(target_space)
    phase = 0.0 if seed is None else (seed % 360) * math.pi / 180.0
    def gen() -> Iterator[int]:
        i = 0
        while True:
            y = (math.sin((i / 8.0) + phase) + 1.0) / 2.0
            yield max(0, min(size - 1, int(round(y * (size - 1)))))
            i += 1
    return StreamSpec("sine_quantized", target_space, size, True, seed, "round(quantized sine)", "structured", gen)


def markov_chain(seed: int | None = None, target_space: str = "digits") -> StreamSpec:
    size = _space_size(target_space)
    def gen() -> Iterator[int]:
        r = _rng(seed, "markov_chain")
        x = r.randrange(size)
        while True:
            yield x
            roll = r.random()
            if roll < 0.72:
                x = (x + 1) % size
            elif roll < 0.88:
                x = x
            else:
                x = r.randrange(size)
    return StreamSpec("markov_chain", target_space, size, True, seed, "transition-biased Markov chain", "structured", gen)


def increment_mod(seed: int | None = None, target_space: str = "integers") -> StreamSpec:
    size = _space_size(target_space)
    start = 0 if seed is None else seed % size
    def gen() -> Iterator[int]:
        i = start
        while True:
            yield i
            i = (i + 1) % size
    return StreamSpec("increment_mod", target_space, size, True, seed, "(n+1) mod K", "structured", gen)


def biased_bit(seed: int | None = None) -> StreamSpec:
    def gen() -> Iterator[int]:
        r = _rng(seed, "biased_bit")
        while True:
            yield 1 if r.random() < 0.65 else 0
    return StreamSpec("biased_bit", "bits", 2, True, seed, "Bernoulli(p=0.65)", "bias", gen)


def biased_digit(seed: int | None = None) -> StreamSpec:
    def gen() -> Iterator[int]:
        r = _rng(seed, "biased_digit")
        while True:
            yield 7 if r.random() < 0.24 else r.choice([0, 1, 2, 3, 4, 5, 6, 8, 9])
    return StreamSpec("biased_digit", "digits", 10, True, seed, "digit 7 overrepresented", "bias", gen)


def weak_lcg_low_bits(seed: int | None = None) -> StreamSpec:
    def gen() -> Iterator[int]:
        state = (seed or 1) & 0x7fffffff
        while True:
            state = (1103515245 * state + 12345) & 0x7fffffff
            yield state & 0xff
    return StreamSpec("weak_lcg_low_bits", "bytes", 256, True, seed, "glibc-style LCG low 8 bits", "bias", gen)


def python_random_pcg(seed: int | None = None, target_space: str = "bytes") -> StreamSpec:
    size = _space_size(target_space)
    def gen() -> Iterator[int]:
        if np is not None:
            rg = np.random.default_rng(seed)
            while True:
                yield int(rg.integers(0, size))
        else:
            r = _rng(seed, "python_random_pcg")
            while True:
                yield r.randrange(size)
    alg = "numpy PCG64 default_rng" if np is not None else "python random fallback"
    return StreamSpec("python_random_pcg", target_space, size, True, seed, alg, "null", gen)


def os_urandom_bytes(seed: int | None = None) -> StreamSpec:
    def gen() -> Iterator[int]:
        while True:
            yield os.urandom(1)[0]
    return StreamSpec("os_urandom_bytes", "bytes", 256, False, None, "os.urandom bytes", "null", gen)


def system_random_digits(seed: int | None = None) -> StreamSpec:
    def gen() -> Iterator[int]:
        r = secrets.SystemRandom()
        while True:
            yield r.randrange(10)
    return StreamSpec("secrets_system_random_digits", "digits", 10, False, None, "secrets.SystemRandom digits", "null", gen)


def suite_streams(suite: str, seed: int | None) -> list[StreamSpec]:
    controls = [repeating_sequence(seed, "digits"), sine_quantized(seed, "digits"), markov_chain(seed, "digits"), increment_mod(seed, "integers")]
    bias = [biased_bit(seed), biased_digit(seed), weak_lcg_low_bits(seed)]
    null = [python_random_pcg(seed, "bytes"), os_urandom_bytes(seed), system_random_digits(seed)]
    if suite == "smoke":
        return [repeating_sequence(seed, "digits"), biased_bit(seed), python_random_pcg(seed, "bytes")]
    if suite == "controls":
        return controls + bias
    if suite == "null":
        return null
    if suite == "full":
        return controls + bias + null
    raise ValueError(f"unknown suite {suite!r}")
