from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventMergeConfig:
    cooldown: int = 24


@dataclass
class EventMerger:
    config: EventMergeConfig
    episode_id: int = 0
    cooldown_remaining: int = 0
    active_episode: int = 0
    episode_age: int = 0

    def update(self, status: str) -> dict[str, int | bool]:
        if status in {"AMBER", "RED"}:
            if self.cooldown_remaining <= 0:
                self.episode_id += 1
                self.active_episode = self.episode_id
                self.episode_age = 0
            self.cooldown_remaining = self.config.cooldown
            self.episode_age += 1
            return {
                "event_active": True,
                "episode_id": self.active_episode,
                "episode_age": self.episode_age,
                "merge_cooldown_remaining": self.cooldown_remaining,
            }

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.episode_age += 1
            return {
                "event_active": True,
                "episode_id": self.active_episode,
                "episode_age": self.episode_age,
                "merge_cooldown_remaining": self.cooldown_remaining,
            }

        self.active_episode = 0
        self.episode_age = 0
        return {
            "event_active": False,
            "episode_id": 0,
            "episode_age": 0,
            "merge_cooldown_remaining": 0,
        }
