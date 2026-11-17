"""Federated-loop configuration (torch-free)."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from afkn.sim.config import ConfigError, build_dataclass, deep_merge


@dataclass(frozen=True)
class FedSpec:
    """Round structure. Project defaults: 100 rounds, 2 local epochs, 60% participation."""

    rounds: int = 100
    local_epochs: int = 2
    participation: float = 0.6
    server_lr: float = 1.0
    eval_every: int = 1
    weight_floor: float = 0.0

    def __post_init__(self) -> None:
        if self.rounds < 1 or self.local_epochs < 1 or self.eval_every < 1:
            raise ConfigError("rounds, local_epochs and eval_every must be >= 1")
        if not 0.0 < self.participation <= 1.0:
            raise ConfigError("participation must be in (0, 1]")
        if self.server_lr <= 0:
            raise ConfigError("server_lr must be > 0")
        if not 0.0 <= self.weight_floor < 1.0:
            raise ConfigError("weight_floor must be in [0, 1)")


def load_fed_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> FedSpec:
    """Load from YAML (default: packaged ``fed.yaml``) with optional deep-merged overrides."""
    if path is None:
        text = resources.files("afkn.configs").joinpath("fed.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if overrides:
        raw = deep_merge(raw, overrides)
    return build_dataclass(FedSpec, raw, "fed")
