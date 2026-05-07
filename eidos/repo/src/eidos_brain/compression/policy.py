"""Configurable policy for anomaly-preserving residual compression."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping


STATUS_ORDER = ("GREEN", "BLUE", "VIOLET", "AMBER", "RED")


@dataclass(frozen=True)
class CompressionRule:
    """Compression behavior for a Sentinel regime."""

    mode: str
    quantization_scale: float
    residual_threshold: float
    surprise_threshold: float
    allow_null: bool = False
    preserve_residual: bool = False
    preserve_raw_frame: bool = False
    preserve_feature_structure: bool = False
    include_model_state: bool = False
    include_sentinel_metrics: bool = True
    description: str = ""


@dataclass(frozen=True)
class CompressionDecision:
    """Resolved policy decision for a single frame."""

    status: str
    mode: str
    quantization_scale: float
    allow_null: bool
    preserve_residual: bool
    preserve_raw_frame: bool
    preserve_feature_structure: bool
    include_model_state: bool
    include_sentinel_metrics: bool
    reason: str


def _default_rules() -> dict[str, CompressionRule]:
    return {
        "GREEN": CompressionRule(
            mode="reference_or_null",
            quantization_scale=0.05,
            residual_threshold=0.05,
            surprise_threshold=1.5,
            allow_null=True,
            description="Aggressive reference/null compression for ordinary frames.",
        ),
        "BLUE": CompressionRule(
            mode="low_residual",
            quantization_scale=0.025,
            residual_threshold=0.25,
            surprise_threshold=2.5,
            preserve_residual=True,
            description="Moderate quantized residual for mild departures.",
        ),
        "VIOLET": CompressionRule(
            mode="structured_residual",
            quantization_scale=0.01,
            residual_threshold=0.75,
            surprise_threshold=4.0,
            preserve_residual=True,
            preserve_feature_structure=True,
            description="Keep residual structure and anomaly metadata.",
        ),
        "AMBER": CompressionRule(
            mode="anomaly_capsule",
            quantization_scale=0.001,
            residual_threshold=1.5,
            surprise_threshold=6.0,
            preserve_residual=True,
            preserve_raw_frame=True,
            preserve_feature_structure=True,
            description="Preserve raw frame context for likely anomalies.",
        ),
        "RED": CompressionRule(
            mode="raw_frame_plus_full_context",
            quantization_scale=0.0,
            residual_threshold=float("inf"),
            surprise_threshold=float("inf"),
            preserve_residual=True,
            preserve_raw_frame=True,
            preserve_feature_structure=True,
            include_model_state=True,
            include_sentinel_metrics=True,
            description="Near-lossless incident frame with full replay context.",
        ),
    }


@dataclass(frozen=True)
class CompressionPolicyConfig:
    """Thresholds and rules used to map residuals/Sentinel status to token modes."""

    rules: Mapping[str, CompressionRule] = field(default_factory=_default_rules)
    null_residual_norm: float = 1e-8
    violet_residual_norm: float = 0.75
    amber_residual_norm: float = 1.5
    violet_surprise_z: float = 4.0
    amber_surprise_z: float = 6.0
    red_surprise_z: float = 9.0
    default_status: str = "GREEN"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None = None) -> "CompressionPolicyConfig":
        """Build a config from a plain dictionary without requiring callers to import dataclasses."""

        if not config:
            return cls()

        base = cls()
        rules = dict(base.rules)
        raw_rules = config.get("rules", {})
        for status, override in raw_rules.items():
            key = normalize_sentinel_status(status, default=status)
            if key not in rules:
                continue
            if isinstance(override, CompressionRule):
                rules[key] = override
            elif isinstance(override, Mapping):
                rules[key] = replace(rules[key], **dict(override))

        scalar_fields = {
            "null_residual_norm",
            "violet_residual_norm",
            "amber_residual_norm",
            "violet_surprise_z",
            "amber_surprise_z",
            "red_surprise_z",
            "default_status",
        }
        values = {name: config[name] for name in scalar_fields if name in config}
        return cls(rules=rules, **values)


def normalize_sentinel_status(status: Any, default: str = "GREEN") -> str:
    """Normalize free-form Sentinel status strings to an Eidos color."""

    if status is None:
        return default
    text = str(status).strip().upper()
    if not text:
        return default
    for color in reversed(STATUS_ORDER):
        if text == color or text.startswith(color) or f" {color}" in text or f"{color}_" in text:
            return color
    if "CALIBRAT" in text:
        return "BLUE"
    return default


class CompressionPolicy:
    """Resolve compression modes from Sentinel status, residual norm, and surprise."""

    def __init__(self, config: CompressionPolicyConfig | Mapping[str, Any] | None = None) -> None:
        self.config = (
            CompressionPolicyConfig.from_mapping(config)
            if isinstance(config, Mapping)
            else config
            if config is not None
            else CompressionPolicyConfig()
        )

    def decide(self, sentinel_status: Any, residual_norm: float, surprise_z: float | None = None) -> CompressionDecision:
        status = normalize_sentinel_status(sentinel_status, default=self.config.default_status)
        surprise = float(surprise_z or 0.0)
        residual = float(residual_norm)
        reason = f"sentinel={status}"

        if surprise >= self.config.red_surprise_z:
            status = "RED"
            reason = f"surprise_z>={self.config.red_surprise_z:g}"
        elif status not in {"AMBER", "RED"} and (
            residual >= self.config.amber_residual_norm or surprise >= self.config.amber_surprise_z
        ):
            status = "AMBER"
            reason = "configured amber residual/surprise threshold"
        elif status not in {"VIOLET", "AMBER", "RED"} and (
            residual >= self.config.violet_residual_norm or surprise >= self.config.violet_surprise_z
        ):
            status = "VIOLET"
            reason = "configured violet residual/surprise threshold"

        rule = self.config.rules.get(status) or self.config.rules[self.config.default_status]
        allow_null = bool(rule.allow_null and residual <= self.config.null_residual_norm and status == "GREEN")
        return CompressionDecision(
            status=status,
            mode=rule.mode,
            quantization_scale=float(rule.quantization_scale),
            allow_null=allow_null,
            preserve_residual=bool(rule.preserve_residual),
            preserve_raw_frame=bool(rule.preserve_raw_frame),
            preserve_feature_structure=bool(rule.preserve_feature_structure),
            include_model_state=bool(rule.include_model_state),
            include_sentinel_metrics=bool(rule.include_sentinel_metrics),
            reason=reason,
        )
