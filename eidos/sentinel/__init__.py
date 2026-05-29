"""Sentinel confirmation layer for proof-plan false-positive control."""

from .calibration import MODE_CONFIGS, MODE_NAMES, SentinelModeConfig, get_mode_config
from .event_merge import ConfirmedEvent, event_to_incident_card, merge_confirmed_events
from .hysteresis import ConfirmationResult, EvidenceFrame, SentinelEventConfirmer, process_stream

__all__ = [
    "MODE_CONFIGS",
    "MODE_NAMES",
    "SentinelModeConfig",
    "get_mode_config",
    "ConfirmedEvent",
    "event_to_incident_card",
    "merge_confirmed_events",
    "ConfirmationResult",
    "EvidenceFrame",
    "SentinelEventConfirmer",
    "process_stream",
]
