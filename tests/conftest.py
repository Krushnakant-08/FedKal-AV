"""Shared fixtures. Every test builds its own data from a seed; nothing is cached on disk."""

from __future__ import annotations

import pytest

from afkn.sim.config import SimConfig, SplitSpec, load_config


@pytest.fixture(scope="session")
def default_cfg() -> SimConfig:
    """The packaged default configuration (spec-sized: 1000 steps, 10 clients)."""
    return load_config()


@pytest.fixture(scope="session")
def small_cfg(default_cfg: SimConfig) -> SimConfig:
    """Same fleet, tiny sizes, for fast structural tests."""
    return default_cfg.replace(steps=150, splits=SplitSpec(train=6, val=3, test=3, global_test=10, trusted=10))
