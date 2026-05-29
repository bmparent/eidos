from __future__ import annotations

from dataclasses import dataclass

from .calibration import finite_scalar


@dataclass(frozen=True)
class HysteresisConfig:
    rise: float = 0.58
    decay: float = 0.90
    amber_threshold: float = 1.0
    red_threshold: float = 2.25


@dataclass
class EventHysteresis:
    config: HysteresisConfig
    event_score: float = 0.0

    def update(self, evidence: float) -> float:
        evidence = max(0.0, finite_scalar(evidence))
        if evidence > 0.0:
            self.event_score = self.event_score * self.config.decay + self.config.rise * evidence
        else:
            self.event_score = self.event_score * self.config.decay
        self.event_score = finite_scalar(max(0.0, self.event_score))
        return self.event_score
