from eidos_agents.budget import BudgetManager
from eidos_agents.pricing import ModelPrice, PriceRegistry
from eidos_agents.routed_contracts import SDKSentryWorkOrder
from eidos_agents.schemas import (
    ArchivistResult,
    BudgetSpec,
    EvidenceItem,
    EvidencePacket,
    EvidenceSourceType,
)
from eidos_agents.sdk_runtime import LiveTelemetry
from decimal import Decimal


def _prices():
    return PriceRegistry(
        "test",
        {"cheap": ModelPrice(Decimal(1), Decimal("0.1"), Decimal(2), None)},
    )


def test_sentry_work_order_canonicalizes_namespaced_archivist_refs():
    order = SDKSentryWorkOrder(
        task_id="TASK",
        objective="assess evidence",
        questions=["what is supported?"],
        allowed_scope=["**"],
        forbidden_scope=[],
        evidence_refs=[
            "base_commit:abc",
            "Archivist:E1",
            "Archivist:E2=path/to/receipt.json",
        ],
        required_output="SDKSentryResult",
        success_criteria=["cite evidence"],
    )
    assert order.evidence_refs == ["base_commit:abc", "E1", "E2"]


def test_canonical_sentry_ref_satisfies_strict_packet_provenance_gate():
    telemetry = LiveTelemetry(
        budget=BudgetManager("TASK", BudgetSpec(), _prices()),
        required_sequence=["archivist", "sentry", "curie"],
        task_id="TASK",
    )
    packet = EvidencePacket(
        task_id="TASK",
        question="q",
        evidence=[
            EvidenceItem(
                evidence_id="E1",
                source_type=EvidenceSourceType.PROJECT_EVIDENCE,
                source="repo",
                location="receipt.json",
                claim_supported="receipt exists",
                excerpt_or_summary="bounded receipt",
                confidence=0.9,
            )
        ],
    )
    telemetry.agent_outputs.append(
        (
            "archivist",
            ArchivistResult(
                task_id="TASK",
                agent="archivist",
                status="PASS",
                summary="packet",
                confidence=0.9,
                evidence_packet=packet,
            ),
        )
    )
    from eidos_agents.schemas import AgentResult

    telemetry.verify_output(
        "sentry",
        AgentResult(
            task_id="TASK",
            agent="sentry",
            status="INCONCLUSIVE",
            summary="current claim not established",
            evidence_refs=["E1"],
            confidence=0.8,
        ),
    )
