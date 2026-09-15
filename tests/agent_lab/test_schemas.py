import pytest
from pydantic import ValidationError

from eidos_agents.schemas import (
    Claim,
    ClaimStatus,
    EvidenceItem,
    EvidenceSourceType,
    Hypothesis,
)


def test_evidence_roundtrip_and_hash_stability():
    item = EvidenceItem(
        evidence_id="E-1", source_type=EvidenceSourceType.PROJECT_EVIDENCE, source="repo", location="x.json",
        claim_supported="receipt exists", excerpt_or_summary="bounded", confidence=0.8,
    )
    assert EvidenceItem.model_validate_json(item.model_dump_json()) == item
    assert item.stable_hash() == EvidenceItem.model_validate_json(item.model_dump_json()).stable_hash()


def test_invalid_enum_rejected():
    with pytest.raises(ValidationError):
        Claim(statement="x", status="PROVED", evidence_refs=[])


def test_required_field_rejected():
    with pytest.raises(ValidationError):
        EvidenceItem.model_validate({"evidence_id": "E-1"})


def test_known_claim_requires_two_evidence_refs():
    with pytest.raises(ValidationError):
        Claim(statement="x", status=ClaimStatus.KNOWN, evidence_refs=["one"])


def test_known_hypothesis_rejects_counterevidence():
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_id="H-1", statement="x", mechanism="m", alternatives=[], predictions=[], falsifiers=[],
            evidence_for=["e1", "e2"], evidence_against=["e3"], status=ClaimStatus.KNOWN,
        )
