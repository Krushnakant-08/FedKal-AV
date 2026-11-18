"""Experiment configuration (torch-free): scales, attack strengths and the fairness audit thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from afkn.sim.config import ConfigError, deep_merge

SCALE_KEYS = {
    "steps", "splits", "rounds", "local_epochs", "eval_every", "seeds", "kalmannet", "noise_levels",
    "hetero_levels", "faulty_fracs", "participations", "attack_fracs", "attacks", "ablation_variants",
    "ablation_attacks", "ablation_attack_frac",
}
HETERO_LEVELS = ("homog", "mild", "strong")


@dataclass(frozen=True)
class ScaleSpec:
    steps: int
    splits: dict[str, int]
    rounds: int
    local_epochs: int
    eval_every: int
    seeds: tuple[int, ...]
    kalmannet: dict[str, Any]
    noise_levels: tuple[float, ...]
    hetero_levels: tuple[str, ...]
    faulty_fracs: tuple[float, ...]
    participations: tuple[float, ...]
    attack_fracs: tuple[float, ...]
    attacks: tuple[str, ...]
    ablation_variants: tuple[str, ...]
    ablation_attacks: tuple[str, ...]
    ablation_attack_frac: float

    def __post_init__(self) -> None:
        if self.steps < 2 or self.rounds < 1 or self.local_epochs < 1 or self.eval_every < 1:
            raise ConfigError("steps >= 2, rounds, local_epochs and eval_every >= 1")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ConfigError("seeds must be non-empty and distinct")
        if set(self.splits) != {"train", "val", "test", "global_test", "trusted"}:
            raise ConfigError("splits needs train, val, test, global_test, trusted")
        bad = [h for h in self.hetero_levels if h not in HETERO_LEVELS]
        if bad:
            raise ConfigError(f"unknown hetero levels {bad}; expected {HETERO_LEVELS}")


@dataclass(frozen=True)
class FairnessSpec:
    low_profiles: tuple[str, ...] = ("low_grade",)
    reference_profiles: tuple[str, ...] = ("high_grade", "mid_grade")
    min_relative_weight_ratio: float = 0.5


@dataclass(frozen=True)
class EvalConfig:
    scale_name: str
    scale: ScaleSpec
    attack_strengths: dict[str, dict[str, float]] = field(default_factory=dict)
    fairness: FairnessSpec = field(default_factory=FairnessSpec)


def _tuple(v: Any) -> Any:
    return tuple(v) if isinstance(v, list) else v


def load_eval_config(scale: str = "smoke", path: str | Path | None = None,
                     overrides: dict[str, Any] | None = None) -> EvalConfig:
    """Load a named scale from ``experiments.yaml``; ``overrides`` is deep-merged into that scale's block."""
    text = (resources.files("afkn.configs").joinpath("experiments.yaml").read_text(encoding="utf-8")
            if path is None else Path(path).read_text(encoding="utf-8"))
    raw = yaml.safe_load(text)
    if scale not in raw["scales"]:
        raise ConfigError(f"unknown scale '{scale}'; expected one of {sorted(raw['scales'])}")
    block = raw["scales"][scale]
    if overrides:
        block = deep_merge(block, overrides)
    unknown = set(block) - SCALE_KEYS
    if unknown:
        raise ConfigError(f"unknown scale keys: {sorted(unknown)}")
    kwargs = {k: _tuple(v) for k, v in block.items()}
    fair = raw.get("fairness", {})
    allowed = {f.name for f in fields(FairnessSpec)}
    if set(fair) - allowed:
        raise ConfigError(f"unknown fairness keys: {sorted(set(fair) - allowed)}")
    return EvalConfig(
        scale_name=scale,
        scale=ScaleSpec(**kwargs),
        attack_strengths={k: dict(v) for k, v in raw.get("attack_strengths", {}).items()},
        fairness=FairnessSpec(**{k: _tuple(v) for k, v in fair.items()}),
    )
