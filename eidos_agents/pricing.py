"""Versioned estimated-cost accounting, kept separate from routing decisions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: Decimal | None
    cached_input_per_million: Decimal | None
    output_per_million: Decimal | None
    reasoning_per_million: Decimal | None


class PriceRegistry:
    def __init__(
        self,
        version: str,
        prices: dict[str, ModelPrice],
        notes: list[str] | None = None,
        *,
        effective_date: str | None = None,
        sources: list[str] | None = None,
    ) -> None:
        self.version = version
        self.effective_date = effective_date
        self.prices = prices
        self.notes = notes or []
        self.sources = sources or []

    @classmethod
    def load(cls, path: Path) -> PriceRegistry:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        prices: dict[str, ModelPrice] = {}
        for model, value in data.get("models", {}).items():
            def dec(key: str, entry: dict[str, object] = value) -> Decimal | None:
                raw = entry.get(key)
                return None if raw is None else Decimal(str(raw))

            prices[model] = ModelPrice(dec("input_per_million"), dec("cached_input_per_million"), dec("output_per_million"), dec("reasoning_per_million"))
        return cls(
            str(data["version"]),
            prices,
            list(data.get("notes", [])),
            effective_date=str(data["effective_date"]) if data.get("effective_date") else None,
            sources=list(data.get("sources", [])),
        )

    def estimate(self, model: str, usage: dict[str, int | None]) -> Decimal | None:
        price = self.prices.get(model)
        if not price or price.input_per_million is None or price.output_per_million is None:
            return None
        cached = int(usage.get("cached_input_tokens") or 0)
        total_in = int(usage.get("input_tokens") or 0)
        output = int(usage.get("output_tokens") or 0)
        reasoning = int(usage.get("reasoning_tokens") or 0)
        regular = max(0, total_in - cached)
        amount = Decimal(regular) * price.input_per_million
        amount += Decimal(cached) * (price.cached_input_per_million or price.input_per_million)
        amount += Decimal(output) * price.output_per_million
        if reasoning and price.reasoning_per_million is not None:
            amount += Decimal(reasoning) * price.reasoning_per_million
        return amount / Decimal(1_000_000)
