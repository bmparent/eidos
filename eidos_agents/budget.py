"""Deterministic task, agent, call, retry, and model cost limits."""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal

from .pricing import PriceRegistry
from .schemas import BudgetSpec, CostReceipt


class BudgetExceeded(RuntimeError):
    pass


class BudgetManager:
    def __init__(self, task_id: str, spec: BudgetSpec, prices: PriceRegistry) -> None:
        self.task_id = task_id
        self.spec = spec
        self.prices = prices
        self.calls_by_model: Counter[str] = Counter()
        self.calls_by_agent: Counter[str] = Counter()
        self.tool_calls: Counter[str] = Counter()
        self.tokens: Counter[str] = Counter()
        self.agent_cost: dict[str, Decimal] = defaultdict(Decimal)
        self.total_cost = Decimal(0)
        self.unknown_cost_calls: list[str] = []

    def authorize_call(self, agent: str, model: str, *, council: bool = False) -> None:
        if council and not self.spec.council_enabled:
            raise BudgetExceeded("Council is disabled")
        if sum(self.calls_by_agent.values()) >= self.spec.maximum_specialist_calls:
            raise BudgetExceeded("maximum specialist calls reached")
        limit = self.spec.per_agent_budget_usd.get(agent)
        if limit is not None and self.agent_cost[agent] >= Decimal(str(limit)):
            raise BudgetExceeded(f"per-agent budget exhausted for {agent}")
        if self.spec.task_budget_usd is not None and self.total_cost >= Decimal(str(self.spec.task_budget_usd)):
            raise BudgetExceeded("task budget exhausted")

    def record_call(self, agent: str, model: str, usage: dict[str, int | None]) -> None:
        self.calls_by_agent[agent] += 1
        self.calls_by_model[model] += 1
        for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"):
            self.tokens[key] += int(usage.get(key) or 0)
        estimated = self.prices.estimate(model, usage)
        if estimated is None:
            self.unknown_cost_calls.append(model)
        else:
            self.total_cost += estimated
            self.agent_cost[agent] += estimated
        if self.spec.task_budget_usd is not None and self.total_cost > Decimal(str(self.spec.task_budget_usd)):
            raise BudgetExceeded("task budget exceeded by completed call")

    def record_tool(self, tool: str) -> None:
        self.tool_calls[tool] += 1

    def receipt(self) -> CostReceipt:
        budget = None if self.spec.task_budget_usd is None else float(Decimal(str(self.spec.task_budget_usd)) - self.total_cost)
        notes = list(self.prices.notes)
        if self.unknown_cost_calls:
            notes.append("No configured price for: " + ", ".join(sorted(set(self.unknown_cost_calls))))
        return CostReceipt(
            task_id=self.task_id,
            price_version=self.prices.version,
            price_effective_date=self.prices.effective_date,
            total_input_tokens=self.tokens["input_tokens"],
            cached_input_tokens=self.tokens["cached_input_tokens"],
            output_tokens=self.tokens["output_tokens"],
            reasoning_tokens=self.tokens["reasoning_tokens"],
            calls_by_model=dict(self.calls_by_model),
            calls_by_agent=dict(self.calls_by_agent),
            tool_calls=dict(self.tool_calls),
            approximate_cost_usd=None if self.unknown_cost_calls else float(self.total_cost),
            budget_remaining=budget,
            estimate_notes=notes,
        )
