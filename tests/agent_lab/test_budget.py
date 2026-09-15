from decimal import Decimal

import pytest

from eidos_agents.budget import BudgetExceeded, BudgetManager
from eidos_agents.pricing import ModelPrice, PriceRegistry
from eidos_agents.schemas import BudgetSpec


def prices():
    return PriceRegistry("test", {"cheap": ModelPrice(Decimal(1), Decimal("0.1"), Decimal(2), None)})


def test_budget_counts_cached_tokens_without_hiding_them():
    budget = BudgetManager("T", BudgetSpec(task_budget_usd=1), prices())
    budget.record_call("archivist", "cheap", {"input_tokens": 1000, "cached_input_tokens": 500, "output_tokens": 100})
    receipt = budget.receipt()
    assert receipt.cached_input_tokens == 500
    assert receipt.approximate_cost_usd is not None


def test_specialist_call_limit_blocks():
    budget = BudgetManager("T", BudgetSpec(maximum_specialist_calls=1), prices())
    budget.record_call("archivist", "cheap", {})
    with pytest.raises(BudgetExceeded):
        budget.authorize_call("curie", "cheap")


def test_council_disabled_by_default():
    budget = BudgetManager("T", BudgetSpec(), prices())
    with pytest.raises(BudgetExceeded):
        budget.authorize_call("council", "expensive", council=True)


def test_unknown_pricing_is_reported_not_faked():
    budget = BudgetManager("T", BudgetSpec(), prices())
    budget.record_call("director", "unknown", {"input_tokens": 10})
    receipt = budget.receipt()
    assert receipt.approximate_cost_usd is None
    assert any("unknown" in note for note in receipt.estimate_notes)
