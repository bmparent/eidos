"""Deterministic task, agent, call, retry, and model cost limits."""

from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from .pricing import PriceRegistry
from .schemas import BudgetSpec, CostReceipt


class BudgetExceeded(RuntimeError):
    pass


class DailyBudgetLedger:
    """Cross-run daily cost ledger backed by SQLite for process-safe persistence."""

    def __init__(self, path: Path, limit_usd: Decimal) -> None:
        self.path = path.resolve()
        self.limit_usd = limit_usd
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS charges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    day_utc TEXT NOT NULL,
                    charged_at TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    model TEXT NOT NULL,
                    cost_usd TEXT NOT NULL,
                    usage_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_daily_charges_day ON charges(day_utc)"
            )

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def day_utc() -> str:
        return datetime.now(UTC).date().isoformat()

    def spent_today(self) -> Decimal:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT cost_usd FROM charges WHERE day_utc=?", (self.day_utc(),)
            ).fetchall()
        return sum((Decimal(row[0]) for row in rows), Decimal(0))

    def remaining_today(self) -> Decimal:
        return self.limit_usd - self.spent_today()

    def record(
        self,
        *,
        task_id: str,
        agent: str,
        model: str,
        cost_usd: Decimal,
        usage: dict[str, int | None],
    ) -> Decimal:
        now = datetime.now(UTC)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO charges(day_utc,charged_at,task_id,agent,model,cost_usd,usage_json)
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    now.date().isoformat(),
                    now.isoformat(),
                    task_id,
                    agent,
                    model,
                    str(cost_usd),
                    json.dumps(usage, sort_keys=True),
                ),
            )
            rows = conn.execute(
                "SELECT cost_usd FROM charges WHERE day_utc=?", (now.date().isoformat(),)
            ).fetchall()
        return sum((Decimal(row[0]) for row in rows), Decimal(0))


class BudgetManager:
    def __init__(
        self,
        task_id: str,
        spec: BudgetSpec,
        prices: PriceRegistry,
        *,
        daily_budget_usd: float | Decimal | None = None,
        daily_ledger_path: Path | None = None,
    ) -> None:
        self.task_id = task_id
        self.spec = spec
        self.prices = prices
        self.calls_by_model: Counter[str] = Counter()
        self.calls_by_agent: Counter[str] = Counter()
        self.specialist_invocations: Counter[str] = Counter()
        self.tool_calls: Counter[str] = Counter()
        self.tokens: Counter[str] = Counter()
        self.agent_cost: dict[str, Decimal] = defaultdict(Decimal)
        self.total_cost = Decimal(0)
        self.unknown_cost_calls: list[str] = []

        env_limit = os.getenv("EIDOS_AGENT_DAILY_BUDGET_USD")
        resolved_limit = (
            Decimal(str(daily_budget_usd))
            if daily_budget_usd is not None
            else (Decimal(env_limit) if env_limit else None)
        )
        if resolved_limit is not None:
            ledger_env = os.getenv("EIDOS_AGENT_DAILY_LEDGER_DB")
            ledger_path = daily_ledger_path or Path(
                ledger_env or "artifacts/agent_lab/daily_budget.sqlite"
            )
            self.daily_ledger: DailyBudgetLedger | None = DailyBudgetLedger(
                ledger_path, resolved_limit
            )
        else:
            self.daily_ledger = None

    def authorize_call(self, agent: str, model: str, *, council: bool = False) -> None:
        if council and not self.spec.council_enabled:
            raise BudgetExceeded("Council is disabled")
        limit = self.spec.per_agent_budget_usd.get(agent)
        if limit is not None and self.agent_cost[agent] >= Decimal(str(limit)):
            raise BudgetExceeded(f"per-agent budget exhausted for {agent}")
        if self.spec.task_budget_usd is not None and self.total_cost >= Decimal(
            str(self.spec.task_budget_usd)
        ):
            raise BudgetExceeded("task budget exhausted")
        if self.daily_ledger is not None and self.daily_ledger.remaining_today() <= 0:
            raise BudgetExceeded("daily Agent Lab budget exhausted")

    def authorize_specialist(
        self, agent: str, model: str, *, council: bool = False
    ) -> None:
        self.authorize_call(agent, model, council=council)
        if (
            sum(self.specialist_invocations.values())
            >= self.spec.maximum_specialist_calls
        ):
            raise BudgetExceeded("maximum specialist calls reached")
        self.specialist_invocations[agent] += 1

    def record_call(
        self, agent: str, model: str, usage: dict[str, int | None]
    ) -> None:
        self.calls_by_agent[agent] += 1
        self.calls_by_model[model] += 1
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
        ):
            self.tokens[key] += int(usage.get(key) or 0)
        estimated = self.prices.estimate(model, usage)
        if estimated is None:
            self.unknown_cost_calls.append(model)
        else:
            self.total_cost += estimated
            self.agent_cost[agent] += estimated
            if self.daily_ledger is not None:
                daily_spend = self.daily_ledger.record(
                    task_id=self.task_id,
                    agent=agent,
                    model=model,
                    cost_usd=estimated,
                    usage=usage,
                )
                if daily_spend > self.daily_ledger.limit_usd:
                    raise BudgetExceeded("daily Agent Lab budget exceeded by completed call")
        if self.spec.task_budget_usd is not None and self.total_cost > Decimal(
            str(self.spec.task_budget_usd)
        ):
            raise BudgetExceeded("task budget exceeded by completed call")

    def record_tool(self, tool: str) -> None:
        self.tool_calls[tool] += 1

    def receipt(self) -> CostReceipt:
        budget = (
            None
            if self.spec.task_budget_usd is None
            else float(Decimal(str(self.spec.task_budget_usd)) - self.total_cost)
        )
        notes = list(self.prices.notes)
        if self.unknown_cost_calls:
            notes.append(
                "No configured price for: "
                + ", ".join(sorted(set(self.unknown_cost_calls)))
            )
        if self.daily_ledger is not None:
            spent = self.daily_ledger.spent_today()
            notes.append(
                "Daily Agent Lab budget: "
                f"${self.daily_ledger.limit_usd}; UTC spend recorded: ${spent}; "
                f"remaining: ${self.daily_ledger.limit_usd - spent}"
            )
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
            approximate_cost_usd=(
                None if self.unknown_cost_calls else float(self.total_cost)
            ),
            budget_remaining=budget,
            estimate_notes=notes,
        )
