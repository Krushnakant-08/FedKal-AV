"""Filter-specific configuration (``afkn/configs/filters.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from afkn.sim.config import ConfigError, build_dataclass, deep_merge


@dataclass(frozen=True)
class UKFSpec:
    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ConfigError("ukf.alpha must be in (0, 1]")


@dataclass(frozen=True)
class ESKFSpec:
    estimate_bias: bool = True
    bias_init_std: tuple[float, float] = (0.02, 0.2)
    bias_rw_std: tuple[float, float] = (0.001, 0.01)

    def __post_init__(self) -> None:
        if min(*self.bias_init_std, *self.bias_rw_std) < 0:
            raise ConfigError("eskf bias stds must be >= 0")


@dataclass(frozen=True)
class FilterConfig:
    ukf: UKFSpec = field(default_factory=UKFSpec)
    eskf: ESKFSpec = field(default_factory=ESKFSpec)


def load_filter_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> FilterConfig:
    """Load ``FilterConfig`` from YAML (default: packaged ``filters.yaml``) with optional overrides."""
    if path is None:
        text = resources.files("afkn.configs").joinpath("filters.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if overrides:
        raw = deep_merge(raw, overrides)
    return build_dataclass(FilterConfig, raw, "filters")
