"""Attack configuration (torch-free)."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from afkn.sim.config import ConfigError, build_dataclass, deep_merge

ATTACKS: tuple[str, ...] = ("none", "random", "sign_flip", "scaling", "bias", "adaptive")


@dataclass(frozen=True)
class AttackSpec:
    attack: str = "none"
    fraction: float = 0.3
    ids: tuple[int, ...] | None = None
    start_round: int = 0
    flip_k: float = 1.0
    scale: float = 10.0
    random_gain: float = 1.0
    bias_gain: float = 1.0
    adaptive_beta: float = 0.3
    adaptive_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.attack not in ATTACKS:
            raise ConfigError(f"attack must be one of {ATTACKS}")
        if not 0.0 <= self.fraction <= 1.0 or self.start_round < 0:
            raise ConfigError("fraction must be in [0, 1] and start_round >= 0")
        if self.ids is not None and (len(set(self.ids)) != len(self.ids) or min(self.ids, default=0) < 0):
            raise ConfigError("ids must be distinct non-negative integers")
        if min(self.flip_k, self.scale, self.random_gain, self.bias_gain, self.adaptive_beta, self.adaptive_scale) < 0:
            raise ConfigError("attack strengths must be >= 0")


def load_attack_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AttackSpec:
    if path is None:
        text = resources.files("afkn.configs").joinpath("attacks.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if overrides:
        raw = deep_merge(raw, overrides)
    return build_dataclass(AttackSpec, raw, "attacks")
