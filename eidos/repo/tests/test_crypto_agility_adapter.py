import numpy as np

from eidos_brain.adapters import CryptoAgilityAdapter, generate_crypto_agility_stream


def test_crypto_adapter_flags_downgrade_and_legacy_crypto():
    adapter = CryptoAgilityAdapter(features=64)
    event = {
        "tls_version": "TLS1.1",
        "previous_tls_version": "TLS1.3",
        "cipher": "TLS_RSA_WITH_3DES_EDE_CBC_SHA",
        "kem_id": "RSA-2048",
        "signature_alg": "ECDSA-SHA1",
        "downgrade_attempt": True,
        "handshake_failures": 30,
        "encrypted_bytes_out": 250_000_000,
        "pqc_failure_count": 7,
    }
    frame, metadata = adapter.transform(event)

    assert frame.shape == (64,)
    assert frame[0] == 1.0
    assert frame[1] == 1.0
    assert frame[4] == 1.0
    assert frame[10] > 0.0
    assert metadata["feature_names"][4] == "downgrade_attempt_indicator"
    assert np.all(np.isfinite(frame))


def test_crypto_downgrade_stream_has_higher_risk_than_normal():
    adapter = CryptoAgilityAdapter(features=64)
    normal_events = generate_crypto_agility_stream("normal", n_frames=32, seed=9)
    risk_events = generate_crypto_agility_stream("crypto_downgrade_exfiltration_risk", n_frames=32, seed=9)
    normal, _ = adapter.transform_many(normal_events)
    adapter = CryptoAgilityAdapter(features=64)
    risky, metadata = adapter.transform_many(risk_events)

    assert any(meta["is_anomaly"] for meta in metadata)
    assert risky[16:, 11].mean() > normal[16:, 11].mean()
