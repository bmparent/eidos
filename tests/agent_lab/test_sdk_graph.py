from eidos_agents.agent_definitions import ForgeWorkspaceAdapter, build_sdk_graph
from eidos_agents.sdk_runtime import _agent_key


def test_director_exposes_all_specialists_as_bounded_tools(lab_config):
    graph = build_sdk_graph(lab_config)
    assert set(graph.tools) == {"archivist", "curie", "sentry", "gauss", "forge", "bench", "auditor", "council"}
    assert graph.director.model == "gpt-5.6-terra"
    assert graph.specialists["forge"].model == "gpt-5.3-codex"
    assert len(graph.director.tools) == 8
    assert {tool.name for tool in graph.specialists["archivist"].tools} == {
        "repo_status",
        "repo_search",
        "repo_read_file",
        "repo_list_tree",
    }
    assert all(
        not graph.specialists[name].tools
        for name in graph.specialists
        if name != "archivist"
    )


def test_council_tool_is_approval_gated(lab_config):
    graph = build_sdk_graph(lab_config)
    assert graph.tools["council"].needs_approval is True


def test_forge_workspace_isolated_behind_adapter():
    description = ForgeWorkspaceAdapter().describe()
    assert description["mode"] == "local-controlled-fallback"


def test_runtime_normalizes_director_and_council_names():
    assert _agent_key("Eidos Director") == "director"
    assert _agent_key("Eidos Council") == "council"
