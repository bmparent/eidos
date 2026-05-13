from eidos_brain.compression.policy import CompressionPolicy, CompressionPolicyConfig, CompressionRule


def test_policy_maps_sentinel_colors_to_modes():
    policy = CompressionPolicy()

    assert policy.decide("GREEN", residual_norm=0.0, surprise_z=0.0).mode == "reference_or_null"
    assert policy.decide("BLUE", residual_norm=0.1, surprise_z=1.8).mode == "low_residual"
    assert policy.decide("VIOLET", residual_norm=0.2, surprise_z=2.0).mode == "structured_residual"
    assert policy.decide("AMBER", residual_norm=0.2, surprise_z=2.0).mode == "anomaly_capsule"
    assert policy.decide("RED", residual_norm=0.2, surprise_z=2.0).mode == "raw_frame_plus_full_context"


def test_policy_thresholds_are_configurable():
    config = CompressionPolicyConfig.from_mapping(
        {
            "amber_residual_norm": 0.2,
            "rules": {
                "AMBER": {
                    "mode": "configured_capsule",
                    "preserve_raw_frame": True,
                    "preserve_residual": True,
                }
            },
        }
    )
    decision = CompressionPolicy(config).decide("GREEN", residual_norm=0.21, surprise_z=0.0)

    assert decision.status == "AMBER"
    assert decision.mode == "configured_capsule"
    assert decision.preserve_raw_frame is True


def test_policy_accepts_direct_rule_objects():
    custom = CompressionPolicyConfig.from_mapping(
        {
            "rules": {
                "BLUE": CompressionRule(
                    mode="tiny_blue",
                    quantization_scale=0.1,
                    residual_threshold=0.1,
                    surprise_threshold=0.5,
                )
            }
        }
    )

    assert CompressionPolicy(custom).decide("CALIBRATING", residual_norm=0.01, surprise_z=0.0).mode == "tiny_blue"
