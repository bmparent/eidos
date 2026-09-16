from decimal import Decimal
from pathlib import Path

import pytest

from eidos_agents.budget import BudgetExceeded, BudgetManager
from eidos_agents.config import DEFAULT_MODELS
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


def test_director_calls_do_not_consume_specialist_call_limit():
    budget = BudgetManager("T", BudgetSpec(maximum_specialist_calls=1), prices())
    budget.record_call("director", "cheap", {})
    budget.authorize_call("archivist", "cheap")
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


def test_production_registry_covers_configured_models_and_records_version():
    registry = PriceRegistry.load(Path("config/agent_lab/model_pricing.yaml"))
    configured_models = {model for model, _reasoning in DEFAULT_MODELS.values()}
    assert configured_models <= registry.prices.keys()
    assert all(registry.prices[model].input_per_million is not None for model in configured_models)
    assert all(registry.prices[model].cached_input_per_million is not None for model in configured_models)
    assert all(registry.prices[model].output_per_million is not None for model in configured_models)

    receipt = BudgetManager("T", BudgetSpec(task_budget_usd=1), registry).receipt()
    assert receipt.price_version == "openai-standard-api-2026-09-15"
    assert receipt.price_effective_date == "2026-09-15"


def test_standard_token_cost_arithmetic_is_exact():
    registry = PriceRegistry.load(Path("config/agent_lab/model_pricing.yaml"))
    cost = registry.estimate(
        "gpt-5.6-terra",
        {
            "input_tokens": 1_000_000,
            "cached_input_tokens": 250_000,
            "output_tokens": 100_000,
        },
    )
    assert cost == Decimal("2.75")


def test_unknown_model_estimate_remains_unknown():
    registry = PriceRegistry.load(Path("config/agent_lab/model_pricing.yaml"))
    assert registry.estimate("not-a-priced-model", {"input_tokens": 1}) is None


def test_production_registry_budget_enforcement_uses_estimated_cost():
    registry = PriceRegistry.load(Path("config/agent_lab/model_pricing.yaml"))
    budget = BudgetManager("T", BudgetSpec(task_budget_usd=1), registry)
    with pytest.raises(BudgetExceeded):
        budget.record_call(
            "director",
            "gpt-5.6-terra",
            {"input_tokens": 500_001, "cached_input_tokens": 0, "output_tokens": 0},
        )
