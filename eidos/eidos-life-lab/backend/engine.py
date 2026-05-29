import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


VERSION = "0.1.0-local"

PATTERNS: Dict[str, List[Tuple[int, int]]] = {
    "glider": [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)],
    "blinker": [(0, 1), (1, 1), (2, 1)],
    "block": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "r_pentomino": [(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)],
    "acorn": [(1, 0), (3, 1), (0, 2), (1, 2), (4, 2), (5, 2), (6, 2)],
    "lightweight_spaceship": [(1, 0), (4, 0), (0, 1), (0, 2), (4, 2), (0, 3), (1, 3), (2, 3), (3, 3)],
    "seed_brush": [(0, 0), (1, 0), (2, 0), (1, 1), (2, 2), (3, 2)],
}

SCENARIOS = {
    "evolutionary_garden",
    "rare_structure_emergence",
    "stress_test",
    "sparse_seed",
    "dense_seed",
}

MUTATION_RATES = {
    "low": 0.001,
    "medium": 0.003,
    "high": 0.008,
    "extreme": 0.02,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class EidosLifeEngine:
    def __init__(self, width: int = 72, height: int = 72, scenario: str = "evolutionary_garden", seed: Optional[int] = None):
        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed if seed is not None else np.random.SeedSequence().entropy % (2**31 - 1))
        self.rng = np.random.default_rng(self.seed)
        self.settings: Dict[str, Any] = {
            "running": False,
            "speed": 1,
            "mutationPressure": "medium",
            "interventionMode": "observational",
            "broadcastFps": 12,
            "eventLogLimit": 180,
            "genomeRegistryCap": 4096,
            "checkpointInterval": 5000,
            "renderQuality": "medium",
        }
        self.generation = 0
        self.scenario = scenario if scenario in SCENARIOS else "evolutionary_garden"
        self.alive = np.zeros((self.height, self.width), dtype=bool)
        self.energy = np.zeros((self.height, self.width), dtype=np.float32)
        self.stress = np.zeros((self.height, self.width), dtype=np.float32)
        self.memory = np.zeros((self.height, self.width), dtype=np.float32)
        self.memory_field = np.zeros((self.height, self.width), dtype=np.float32)
        self.nutrient = np.zeros((self.height, self.width), dtype=np.float32)
        self.waste = np.zeros((self.height, self.width), dtype=np.float32)
        self.signal = np.zeros((self.height, self.width), dtype=np.float32)
        self.genome = np.zeros((self.height, self.width), dtype=np.int32)
        self.lineage = np.zeros((self.height, self.width), dtype=np.int32)
        self.genome_registry: Dict[int, Dict[str, Any]] = {}
        self.event_log: List[Dict[str, Any]] = []
        self._event_cooldowns: Dict[str, int] = {}
        self._genome_counter = 0
        self._lineage_counter = 0
        self._last_alive = 0
        self._last_metrics: Dict[str, Any] = {}
        self.reset(self.scenario)

    def reset(self, scenario: Optional[str] = None) -> Dict[str, Any]:
        if scenario:
            self.scenario = scenario if scenario in SCENARIOS else "evolutionary_garden"
        self.generation = 0
        self.alive.fill(False)
        self.energy.fill(0.0)
        self.stress.fill(0.0)
        self.memory.fill(0.0)
        self.memory_field.fill(0.0)
        self.waste.fill(0.0)
        self.signal.fill(0.0)
        self.genome.fill(0)
        self.lineage.fill(0)
        self.genome_registry.clear()
        self._genome_counter = 0
        self._lineage_counter = 0
        self.event_log.clear()
        self._event_cooldowns.clear()
        self.nutrient[:] = self._initial_nutrient()
        density = {
            "evolutionary_garden": 0.12,
            "rare_structure_emergence": 0.045,
            "stress_test": 0.22,
            "sparse_seed": 0.018,
            "dense_seed": 0.31,
        }.get(self.scenario, 0.12)
        self._seed_random(density)
        if self.scenario == "rare_structure_emergence":
            self.inject_pattern(self.width // 3, self.height // 3, "r_pentomino", log_event=False)
            self.inject_pattern((self.width * 2) // 3, self.height // 2, "acorn", log_event=False)
        elif self.scenario == "sparse_seed":
            self.inject_pattern(self.width // 2, self.height // 2, "acorn", log_event=False)
        self._last_alive = int(self.alive.sum())
        self._log_event("reset", f"Scenario reset to {self.scenario}", {"scenario": self.scenario}, cooldown=0)
        return self.snapshot()

    def apply_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        command = payload.get("command")
        if command == "play":
            self.settings["running"] = True
        elif command == "pause":
            self.settings["running"] = False
        elif command == "step":
            self.step(int(payload.get("steps", 1)))
        elif command == "reset":
            return self.reset(payload.get("scenario"))
        elif command == "set":
            self.update_settings(payload.get("settings") or payload)
        elif command == "toggle_cell":
            self.toggle_cell(int(payload["x"]), int(payload["y"]))
        elif command == "set_cell":
            self.set_cell(int(payload["x"]), int(payload["y"]), bool(payload.get("alive", True)))
        elif command == "paint_disk":
            self.paint_disk(int(payload["x"]), int(payload["y"]), int(payload.get("radius", 2)), payload.get("mode", "birth"))
        elif command == "inject_pattern":
            self.inject_pattern(int(payload["x"]), int(payload["y"]), payload.get("pattern", "glider"))
        elif command == "clear_world":
            self.clear_world()
        elif command == "random_seed":
            self.random_seed(float(payload.get("density", 0.08)))
        elif command == "load_checkpoint":
            self.load_checkpoint(Path(payload["path"]))
        else:
            raise ValueError(f"Unknown command: {command}")
        return self.snapshot()

    def update_settings(self, settings: Dict[str, Any]) -> None:
        if "speed" in settings:
            self.settings["speed"] = int(_clamp(float(settings["speed"]), 1, 64))
        if "mutationPressure" in settings:
            pressure = str(settings["mutationPressure"])
            if pressure in MUTATION_RATES:
                self.settings["mutationPressure"] = pressure
        if "interventionMode" in settings:
            mode = str(settings["interventionMode"])
            self.settings["interventionMode"] = mode if mode in {"observational", "experimental"} else "observational"
        if "broadcastFps" in settings:
            self.settings["broadcastFps"] = int(_clamp(float(settings["broadcastFps"]), 5, 15))
        if "genomeRegistryCap" in settings:
            self.settings["genomeRegistryCap"] = int(_clamp(float(settings["genomeRegistryCap"]), 128, 20000))
        if "renderQuality" in settings:
            quality = str(settings["renderQuality"])
            if quality in {"low", "medium", "high"}:
                self.settings["renderQuality"] = quality

    def step(self, steps: int = 1) -> Dict[str, Any]:
        steps = int(_clamp(steps, 1, 1000))
        for _ in range(steps):
            self._tick()
        return self.snapshot()

    def _tick(self) -> None:
        neighbors = self._neighbor_count(self.alive)
        base_survive = self.alive & ((neighbors == 2) | (neighbors == 3))
        base_birth = (~self.alive) & (neighbors == 3)
        previous_alive = self.alive.copy()
        mutation_pressure = self._mutation_field()

        birth_support = 0.70 + (0.36 * self.nutrient) + (0.14 * self.signal) - (0.45 * self.waste) - (0.22 * self.stress)
        birth_support += mutation_pressure
        birth_support = np.clip(birth_support, 0.05, 0.98)
        births = base_birth & (self.rng.random(self.alive.shape) < birth_support)

        survival_support = 0.95 + (0.12 * self.nutrient) - (0.35 * self.waste) - (0.18 * self.stress)
        survival_support = np.clip(survival_support, 0.20, 0.99)
        survives = base_survive & (self.rng.random(self.alive.shape) < survival_support)

        spontaneous = (~self.alive) & (self.rng.random(self.alive.shape) < np.clip(mutation_pressure * 0.08, 0.0, 0.025))
        births |= spontaneous
        next_alive = survives | births

        deaths = previous_alive & ~next_alive
        self._update_genomes(previous_alive, next_alive, births, spontaneous)
        self.alive = next_alive
        self.generation += 1

        alive_float = self.alive.astype(np.float32)
        death_float = deaths.astype(np.float32)
        change_float = (previous_alive ^ self.alive).astype(np.float32)
        self.nutrient = np.clip(self._diffuse(self.nutrient, 0.16) + 0.018 * (1.0 - alive_float) - 0.050 * alive_float, 0.0, 1.0)
        self.waste = np.clip(self._diffuse(self.waste, 0.12) * 0.96 + 0.018 * alive_float + 0.070 * death_float, 0.0, 1.0)
        self.signal = np.clip(self._diffuse(self.signal, 0.22) * 0.90 + 0.22 * change_float + 0.035 * alive_float, 0.0, 1.0)
        unstable = np.clip(np.abs(neighbors.astype(np.float32) - 2.5) / 5.5, 0.0, 1.0)
        self.stress = np.clip(self._diffuse(self.stress, 0.10) * 0.90 + 0.12 * unstable + 0.22 * self.waste + 0.08 * change_float, 0.0, 1.0)
        self.memory = np.clip(self.memory * 0.982 + 0.055 * alive_float - 0.012 * death_float, 0.0, 1.0)
        self.memory_field = np.clip(self._diffuse(self.memory_field, 0.18) * 0.96 + 0.065 * self.memory, 0.0, 1.0)
        self.energy = np.clip(0.84 * self.energy + 0.16 * (0.55 * self.nutrient + 0.35 * alive_float + 0.18 * self.signal - 0.34 * self.waste - 0.18 * self.stress), 0.0, 1.0)

        metrics = self.metrics(neighbors=neighbors, birth_candidates=base_birth)
        self._detect_events(metrics)
        self._last_alive = int(metrics["alive"])
        self._last_metrics = metrics

    def _update_genomes(self, previous_alive: np.ndarray, next_alive: np.ndarray, births: np.ndarray, spontaneous: np.ndarray) -> None:
        self.genome[~next_alive] = 0
        self.lineage[~next_alive] = 0
        positions = np.argwhere(births)
        mutation_rate = self._base_mutation_rate()
        for y, x in positions:
            parent_genome, parent_lineage = self._choose_parent_genome(int(x), int(y), previous_alive)
            should_mutate = bool(spontaneous[y, x]) or self.rng.random() < mutation_rate
            if parent_genome == 0 or should_mutate:
                lineage_id = parent_lineage if parent_lineage and self.rng.random() > 0.12 else self._next_lineage()
                genome_id = self._next_genome(parent_genome or None, lineage_id)
            else:
                genome_id = parent_genome
                lineage_id = parent_lineage
            self.genome[y, x] = genome_id
            self.lineage[y, x] = lineage_id

    def _choose_parent_genome(self, x: int, y: int, previous_alive: np.ndarray) -> Tuple[int, int]:
        candidates: List[Tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                yy = (y + dy) % self.height
                xx = (x + dx) % self.width
                if previous_alive[yy, xx] and self.genome[yy, xx] > 0:
                    candidates.append((int(self.genome[yy, xx]), int(self.lineage[yy, xx])))
        if not candidates:
            return 0, 0
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def _base_mutation_rate(self) -> float:
        return MUTATION_RATES.get(str(self.settings.get("mutationPressure")), MUTATION_RATES["medium"])

    def _mutation_field(self) -> np.ndarray:
        base = self._base_mutation_rate()
        field = np.full(self.alive.shape, base, dtype=np.float32)
        if self.settings.get("interventionMode") == "experimental":
            field += base * 3.0 * np.clip((0.55 * self.signal) + (0.45 * self.memory_field), 0.0, 1.0)
        return np.clip(field, 0.0, 0.08)

    def toggle_cell(self, x: int, y: int) -> None:
        x, y = self._normalize_xy(x, y)
        self.set_cell(x, y, not bool(self.alive[y, x]), event_name="manual_cell_edit")

    def set_cell(self, x: int, y: int, alive: bool, event_name: str = "manual_cell_edit") -> None:
        x, y = self._normalize_xy(x, y)
        self.alive[y, x] = bool(alive)
        if alive:
            if self.genome[y, x] == 0:
                lineage_id = self._next_lineage()
                self.lineage[y, x] = lineage_id
                self.genome[y, x] = self._next_genome(None, lineage_id)
            self.signal[y, x] = min(1.0, float(self.signal[y, x]) + 0.35)
            self.nutrient[y, x] = max(float(self.nutrient[y, x]), 0.70)
        else:
            self.genome[y, x] = 0
            self.lineage[y, x] = 0
            self.waste[y, x] = min(1.0, float(self.waste[y, x]) + 0.18)
        self._log_event(event_name, f"Cell edit at ({x}, {y})", {"x": x, "y": y, "alive": bool(alive)}, cooldown=0)

    def paint_disk(self, x: int, y: int, radius: int = 2, mode: str = "birth") -> None:
        x, y = self._normalize_xy(x, y)
        radius = int(_clamp(radius, 1, 16))
        edited = 0
        for yy in range(y - radius, y + radius + 1):
            for xx in range(x - radius, x + radius + 1):
                if (xx - x) ** 2 + (yy - y) ** 2 > radius ** 2:
                    continue
                nx, ny = self._normalize_xy(xx, yy)
                if mode == "toggle":
                    self.alive[ny, nx] = not bool(self.alive[ny, nx])
                    if self.alive[ny, nx] and self.genome[ny, nx] == 0:
                        lineage_id = self._next_lineage()
                        self.lineage[ny, nx] = lineage_id
                        self.genome[ny, nx] = self._next_genome(None, lineage_id)
                    elif not self.alive[ny, nx]:
                        self.genome[ny, nx] = 0
                        self.lineage[ny, nx] = 0
                elif mode == "kill":
                    self.alive[ny, nx] = False
                    self.genome[ny, nx] = 0
                    self.lineage[ny, nx] = 0
                    self.waste[ny, nx] = min(1.0, float(self.waste[ny, nx]) + 0.15)
                else:
                    self.alive[ny, nx] = True
                    if self.genome[ny, nx] == 0:
                        lineage_id = self._next_lineage()
                        self.lineage[ny, nx] = lineage_id
                        self.genome[ny, nx] = self._next_genome(None, lineage_id)
                    self.nutrient[ny, nx] = max(float(self.nutrient[ny, nx]), 0.72)
                    self.signal[ny, nx] = min(1.0, float(self.signal[ny, nx]) + 0.20)
                edited += 1
        self._log_event("manual_paint", f"Paint {mode} disk at ({x}, {y})", {"x": x, "y": y, "radius": radius, "mode": mode, "edited": edited}, cooldown=0)

    def inject_pattern(self, x: int, y: int, pattern: str = "glider", log_event: bool = True) -> None:
        pattern = pattern if pattern in PATTERNS else "glider"
        x, y = self._normalize_xy(x, y)
        lineage_id = self._next_lineage()
        edited = 0
        for dx, dy in PATTERNS[pattern]:
            nx, ny = self._normalize_xy(x + dx, y + dy)
            self.alive[ny, nx] = True
            self.lineage[ny, nx] = lineage_id
            self.genome[ny, nx] = self._next_genome(None, lineage_id)
            self.nutrient[ny, nx] = max(float(self.nutrient[ny, nx]), 0.75)
            self.signal[ny, nx] = min(1.0, float(self.signal[ny, nx]) + 0.30)
            edited += 1
        if log_event:
            self._log_event("manual_pattern", f"Injected {pattern}", {"x": x, "y": y, "pattern": pattern, "edited": edited}, cooldown=0)

    def clear_world(self) -> None:
        self.alive.fill(False)
        self.genome.fill(0)
        self.lineage.fill(0)
        self.signal *= 0.5
        self.waste *= 0.65
        self._log_event("manual_pattern", "World cleared", {"alive": 0}, cooldown=0)

    def random_seed(self, density: float) -> None:
        density = _clamp(density, 0.0, 0.55)
        self.alive.fill(False)
        self.genome.fill(0)
        self.lineage.fill(0)
        self._seed_random(density)
        self._log_event("manual_pattern", f"Random seed at density {density:.3f}", {"density": density}, cooldown=0)

    def full_state(self) -> Dict[str, Any]:
        snapshot = self.snapshot()
        return {
            **snapshot,
            "genomeRegistry": self.genome_registry,
            "seed": self.seed,
            "timestamp": utc_now_iso(),
        }

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("width", self.width)) != self.width or int(payload.get("height", self.height)) != self.height:
            raise ValueError("Checkpoint dimensions do not match this engine")
        self.scenario = payload.get("scenario", self.scenario)
        self.settings.update(payload.get("settings", {}))
        self.generation = int(payload.get("generation", 0))
        self.alive = np.array(payload["alive"], dtype=np.uint8).reshape((self.height, self.width)).astype(bool)
        for name, target in [
            ("energy", self.energy),
            ("stress", self.stress),
            ("memory", self.memory),
            ("signal", self.signal),
            ("nutrient", self.nutrient),
            ("waste", self.waste),
            ("memoryField", self.memory_field),
        ]:
            target[:] = np.array(payload[name], dtype=np.float32).reshape((self.height, self.width))
        self.genome = np.array(payload["genome"], dtype=np.int32).reshape((self.height, self.width))
        self.lineage = np.array(payload["lineage"], dtype=np.int32).reshape((self.height, self.width))
        self.genome_registry = {int(k): v for k, v in payload.get("genomeRegistry", {}).items()}
        self._genome_counter = max(self.genome_registry.keys(), default=0)
        self._lineage_counter = int(self.lineage.max(initial=0))
        self._log_event("manual_pattern", f"Loaded checkpoint {path}", {"path": str(path)}, cooldown=0)
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        neighbors = self._neighbor_count(self.alive)
        birth_candidates = (~self.alive) & (neighbors == 3)
        metrics = self.metrics(neighbors=neighbors, birth_candidates=birth_candidates)
        return {
            "version": VERSION,
            "scenario": self.scenario,
            "width": self.width,
            "height": self.height,
            "settings": dict(self.settings),
            "generation": int(self.generation),
            "alive": self.alive.astype(np.uint8).ravel().tolist(),
            "energy": self._float_list(self.energy),
            "stress": self._float_list(self.stress),
            "memory": self._float_list(self.memory),
            "signal": self._float_list(self.signal),
            "nutrient": self._float_list(self.nutrient),
            "waste": self._float_list(self.waste),
            "memoryField": self._float_list(self.memory_field),
            "genome": self.genome.ravel().astype(int).tolist(),
            "lineage": self.lineage.ravel().astype(int).tolist(),
            "metrics": metrics,
            "events": self.event_log[-int(self.settings.get("eventLogLimit", 180)):],
        }

    def metrics(self, neighbors: Optional[np.ndarray] = None, birth_candidates: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if neighbors is None:
            neighbors = self._neighbor_count(self.alive)
        if birth_candidates is None:
            birth_candidates = (~self.alive) & (neighbors == 3)
        alive_count = int(self.alive.sum())
        components, largest = self._components()
        alive_mask = self.alive
        active_genomes = int(len(np.unique(self.genome[alive_mask][self.genome[alive_mask] > 0]))) if alive_count else 0
        active_lineages = int(len(np.unique(self.lineage[alive_mask][self.lineage[alive_mask] > 0]))) if alive_count else 0

        def mean_alive(field: np.ndarray) -> float:
            if not alive_count:
                return 0.0
            return float(np.mean(field[alive_mask]))

        density = alive_count / float(self.width * self.height)
        birth_count = int(birth_candidates.sum())
        sentinel = self._sentinel_regime(alive_count, density, birth_count, largest)
        return {
            "generation": int(self.generation),
            "alive": alive_count,
            "density": round(float(density), 6),
            "components": int(components),
            "largestComponent": int(largest),
            "birthCandidates": birth_count,
            "activeGenomes": active_genomes,
            "activeLineages": active_lineages,
            "aliveEnergyMean": round(mean_alive(self.energy), 6),
            "aliveMemoryMean": round(mean_alive(self.memory), 6),
            "aliveSignalMean": round(mean_alive(self.signal), 6),
            "aliveNutrientMean": round(mean_alive(self.nutrient), 6),
            "aliveWasteMean": round(mean_alive(self.waste), 6),
            "aliveStressMean": round(mean_alive(self.stress), 6),
            "sentinelRegime": sentinel,
        }

    def _sentinel_regime(self, alive_count: int, density: float, birth_candidates: int, largest_component: int) -> str:
        if density < 0.003:
            return "RED_COLLAPSE"
        if density > 0.25:
            return "RED_BLOOM"
        if alive_count and (birth_candidates / alive_count) > 0.75:
            return "AMBER_THRASH"
        if alive_count and (largest_component / alive_count) > 0.25:
            return "AMBER_DOMINANCE"
        return "GREEN_EDGE"

    def _detect_events(self, metrics: Dict[str, Any]) -> None:
        alive = int(metrics["alive"])
        density = float(metrics["density"])
        largest = int(metrics["largestComponent"])
        if largest >= max(96, int(alive * 0.28)):
            self._log_event("large_colony", "Large connected colony detected", {"largestComponent": largest, "alive": alive}, cooldown=160)
        if density > 0.25:
            self._log_event("bloom", "High-density bloom regime", {"density": density}, cooldown=160)
        if density < 0.003:
            self._log_event("collapse", "Low-density collapse regime", {"density": density}, cooldown=80)
        if self._last_alive:
            delta = alive - self._last_alive
            if delta > max(30, self._last_alive * 0.22):
                self._log_event("population_surge", "Population surged sharply", {"previous": self._last_alive, "current": alive}, cooldown=120)
            if delta < -max(30, self._last_alive * 0.22):
                self._log_event("population_drop", "Population dropped sharply", {"previous": self._last_alive, "current": alive}, cooldown=120)
        if alive:
            lineages, counts = np.unique(self.lineage[self.alive], return_counts=True)
            valid = lineages > 0
            if np.any(valid):
                best_idx = int(np.argmax(counts[valid]))
                valid_lineages = lineages[valid]
                valid_counts = counts[valid]
                lineage_id = int(valid_lineages[best_idx])
                count = int(valid_counts[best_idx])
                if count / alive > 0.55 and alive > 40:
                    self._log_event("lineage_takeover", "Lineage takeover candidate", {"lineage": lineage_id, "share": round(count / alive, 4)}, cooldown=220)

    def _log_event(self, kind: str, message: str, data: Optional[Dict[str, Any]] = None, cooldown: int = 60) -> None:
        last_generation = self._event_cooldowns.get(kind, -10**9)
        if cooldown and (self.generation - last_generation) < cooldown:
            return
        self._event_cooldowns[kind] = self.generation
        self.event_log.append(
            {
                "generation": int(self.generation),
                "kind": kind,
                "message": message,
                "data": data or {},
                "timestamp": utc_now_iso(),
            }
        )
        limit = int(self.settings.get("eventLogLimit", 180))
        if len(self.event_log) > limit:
            self.event_log = self.event_log[-limit:]

    def _seed_random(self, density: float) -> None:
        mask = self.rng.random((self.height, self.width)) < density
        self.alive |= mask
        for y, x in np.argwhere(mask):
            lineage_id = self._next_lineage()
            self.lineage[y, x] = lineage_id
            self.genome[y, x] = self._next_genome(None, lineage_id)
        self.energy[mask] = self.rng.uniform(0.35, 0.9, size=int(mask.sum())).astype(np.float32)
        self.memory[mask] = self.rng.uniform(0.05, 0.25, size=int(mask.sum())).astype(np.float32)
        self.signal[mask] = self.rng.uniform(0.04, 0.18, size=int(mask.sum())).astype(np.float32)

    def _initial_nutrient(self) -> np.ndarray:
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        cx = self.width * 0.52
        cy = self.height * 0.48
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        radial = 0.32 + 0.34 * np.exp(-(dist**2) / (2 * (self.width * 0.34) ** 2))
        waves = 0.08 * np.sin(xx / 5.5) + 0.06 * np.cos(yy / 7.0)
        return np.clip(radial + waves + self.rng.normal(0.0, 0.025, size=(self.height, self.width)), 0.05, 0.95).astype(np.float32)

    def _next_lineage(self) -> int:
        self._lineage_counter += 1
        return self._lineage_counter

    def _next_genome(self, parent: Optional[int], lineage_id: int) -> int:
        self._genome_counter += 1
        genome_id = self._genome_counter
        cap = int(self.settings.get("genomeRegistryCap", 4096))
        if len(self.genome_registry) >= cap:
            oldest = sorted(self.genome_registry.keys())[: max(1, len(self.genome_registry) - cap + 1)]
            for key in oldest:
                self.genome_registry.pop(key, None)
        self.genome_registry[genome_id] = {
            "id": genome_id,
            "parent": parent,
            "lineage": lineage_id,
            "createdGeneration": int(self.generation),
            "mutationPressure": self.settings.get("mutationPressure"),
        }
        return genome_id

    def _components(self) -> Tuple[int, int]:
        alive = self.alive
        visited = np.zeros_like(alive, dtype=bool)
        components = 0
        largest = 0
        for start_y, start_x in np.argwhere(alive):
            if visited[start_y, start_x]:
                continue
            components += 1
            size = 0
            queue: deque[Tuple[int, int]] = deque([(int(start_y), int(start_x))])
            visited[start_y, start_x] = True
            while queue:
                y, x = queue.popleft()
                size += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        yy = (y + dy) % self.height
                        xx = (x + dx) % self.width
                        if alive[yy, xx] and not visited[yy, xx]:
                            visited[yy, xx] = True
                            queue.append((yy, xx))
            largest = max(largest, size)
        return components, largest

    def _neighbor_count(self, grid: np.ndarray) -> np.ndarray:
        count = np.zeros_like(grid, dtype=np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                count += np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
        return count

    def _diffuse(self, field: np.ndarray, strength: float) -> np.ndarray:
        neighbor_mean = (
            np.roll(field, 1, axis=0)
            + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
        ) * 0.25
        return field * (1.0 - strength) + neighbor_mean * strength

    def _normalize_xy(self, x: int, y: int) -> Tuple[int, int]:
        return int(x) % self.width, int(y) % self.height

    def _float_list(self, field: np.ndarray) -> List[float]:
        return np.round(field.astype(np.float32).ravel(), 4).tolist()
