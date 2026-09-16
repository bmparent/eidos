import pytest
from pydantic import ValidationError

from eidos_agents.routed_contracts import (
    SDKArchivistWorkOrder,
    SDKCurieWorkOrder,
    SDKSentryResult,
    SDKSentryWorkOrder,
)


def _work_order_payload(required_output: str) -> dict[str, object]:
    return {
        "task_id": "TASK-ROUTED",
        "objective": "bounded routed research",
        "questions": ["what does the evidence support?"],
        "allowed_scope": ["eidos/**"],
        "forbidden_scope": ["apps/**"],
        "evidence_refs": ["E1"],
        "required_output": required_output,
        "success_criteria": ["structured contract returned"],
    }


def test_required_routed_work_orders_pin_exact_output_contracts():
    assert SDKArchivistWorkOrder.model_validate(
        _work_order_payload("SDKArchivistResult")
    ).required_output == "SDKArchivistResult"
    assert SDKSentryWorkOrder.model_validate(
        _work_order_payload("SDKSentryResult")
    ).required_output == "SDKSentryResult"
    assert SDKCurieWorkOrder.model_validate(
        _work_order_payload("SDKCurieResult")
    ).required_output == "SDKCurieResult"


@pytest.mark.parametrize(
    "model,wrong",
    [
        (SDKArchivistWorkOrder, "SpecialistAccounting"),
        (SDKSentryWorkOrder, "EvidencePacket plus SpecialistAccounting plus CostReceipt"),
        (SDKCurieWorkOrder, "RunManifest"),
    ],
)
def test_routed_work_orders_reject_host_owned_or_wrong_output(model, wrong):
    with pytest.raises(ValidationError):
        model.model_validate(_work_order_payload(wrong))


def test_sentry_wire_result_converts_to_generic_persisted_result():
    result = SDKSentryResult(
        task_id="TASK-ROUTED",
        agent="Sentry",
        status="INCONCLUSIVE",
        summary="historical evidence is insufficient for a current behavior change",
        claims=[],
        evidence_refs=["E1"],
        counterevidence_refs=[],
        confidence=0.8,
        risks=["historical receipt is not tied to the current base commit"],
        next_action="commission Curie for the bounded missing experiment",
        recommended_agent="curie",
        deliverables=[],
        integrity_gates=[
            "clean base commit",
            "frozen held-out partition",
            "prediction hash committed before label join",
        ],
        primary_failure_mode="leakage or provenance mismatch creates misleading precision",
        mandatory_detection_guard="reject the run unless provenance and held-out hashes verify",
    )
    internal = result.to_internal()
    assert internal.agent == "Sentry"
    assert internal.evidence_refs == ["E1"]
    assert internal.deliverables["sentry_integrity"]["primary_failure_mode"].startswith(
        "leakage"
    )


def test_sentry_wire_schema_is_closed():
    schema = SDKSentryResult.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["primary_failure_mode"]["type"] == "string"
    assert schema["properties"]["mandatory_detection_guard"]["type"] == "string"
