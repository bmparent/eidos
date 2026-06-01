from __future__ import annotations

from dataclasses import dataclass

from .calibration import finite_scalar


@dataclass(frozen=True)
class NormalSuppressionConfig:
    max_multiplier: float = 1.35
    stable_step: float = 0.015
    unstable_step: float = 0.08
    stable_evidence_ceiling: float = 0.12


@dataclass
class NormalSuppression:
    config: NormalSuppressionConfig
    multiplier_value: float = 1.0

    def update(self, *, residual_evidence: float, input_evidence: float, geometry_evidence: float) -> float:
        worst = max(finite_scalar(residual_evidence), finite_scalar(input_evidence), finite_scalar(geometry_evidence))
        if worst <= self.config.stable_evidence_ceiling:
            self.multiplier_value = min(self.config.max_multiplier, self.multiplier_value + self.config.stable_step)
        else:
            self.multiplier_value = max(1.0, self.multiplier_value - self.config.unstable_step)
        return finite_scalar(self.multiplier_value)
