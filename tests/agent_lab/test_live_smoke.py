import asyncio
import os

import pytest

from eidos_agents.agent_definitions import build_sdk_graph
from eidos_agents.schemas import AgentResult


@pytest.mark.live
def test_opt_in_low_cost_live_archivist_smoke(lab_config):
    if os.getenv("EIDOS_AGENT_LIVE_SMOKE") != "1" or not os.getenv("OPENAI_API_KEY"):
        pytest.skip("requires OPENAI_API_KEY and explicit EIDOS_AGENT_LIVE_SMOKE=1")
    from agents import Runner

    graph = build_sdk_graph(lab_config)
    result = asyncio.run(
        Runner.run(
            graph.specialists["archivist"],
            '{"task_id":"LIVE-SMOKE","objective":"Return an INCONCLUSIVE empty evidence result; do not call tools."}',
            max_turns=1,
        )
    )
    assert isinstance(result.final_output, AgentResult)
