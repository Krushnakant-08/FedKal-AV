"""Layer 2 configuration (torch-free)."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from afkn.sim.config import ConfigError, build_dataclass, deep_merge


@dataclass(frozen=True)
class ConfidenceSpec:
    clip_c: float = 2.0
    perf_scale: float = 0.05
    relative_gain: bool = False
    claim_reported_min: float = 0.05
    claim_measured_max: float = -0.05
    liar_factor: float = 0.5
    combine: str = "product"
    ema_alpha: float = 0.5
    low_score: float = 0.1
    low_rounds: int = 3
    probation_len: int = 5
    weight_floor: float = 0.01
    ref_epochs: int = 2
    use_clip: bool = True
    use_direction: bool = True
    use_perf: bool = True
    use_claim: bool = True
    use_history: bool = True

    def __post_init__(self) -> None:
        if self.clip_c <= 0 or self.perf_scale <= 0:
            raise ConfigError("clip_c and perf_scale must be > 0")
        if self.combine not in ("product", "mean"):
            raise ConfigError("combine must be 'product' or 'mean'")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ConfigError("ema_alpha must be in (0, 1]")
        if not 0.0 <= self.liar_factor <= 1.0 or not 0.0 <= self.low_score <= 1.0:
            raise ConfigError("liar_factor and low_score must be in [0, 1]")
        if self.low_rounds < 1 or self.probation_len < 1 or self.ref_epochs < 1:
            raise ConfigError("low_rounds, probation_len and ref_epochs must be >= 1")
        if not 0.0 <= self.weight_floor < 1.0:
            raise ConfigError("weight_floor must be in [0, 1)")


def load_confidence_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ConfidenceSpec:
    if path is None:
        text = resources.files("afkn.configs").joinpath("confidence.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if overrides:
        raw = deep_merge(raw, overrides)
    return build_dataclass(ConfidenceSpec, raw, "confidence")
