"""Config: YAML loading, validation, overrides, derived matrices."""

from __future__ import annotations

import numpy as np
import pytest

from afkn.sim.config import SEGMENT_TYPES, ConfigError, SimConfig, load_config


def test_default_config_matches_global_spec(default_cfg: SimConfig):
    assert default_cfg.dt == 0.1
    assert default_cfg.steps == 1000
    assert default_cfg.motion_model == "ctrv"
    assert len(default_cfg.clients) == 10
    s = default_cfg.splits
    assert (s.train, s.val, s.test) == (200, 40, 40)
    assert set(default_cfg.segments) == set(SEGMENT_TYPES)


def test_five_profiles_each_used_by_clients(default_cfg: SimConfig):
    assert default_cfg.profile_names == ("high_grade", "mid_grade", "low_grade", "intermittent", "faulty")
    assert {c.profile for c in default_cfg.clients} == set(default_cfg.profile_names)


def test_both_fault_modes_present(default_cfg: SimConfig):
    modes = {c.fault_mode for c in default_cfg.clients if c.profile == "faulty"}
    assert modes == {"constant_bias", "random_spikes"}


def test_Q_and_P0_are_diagonal_matching_config(default_cfg: SimConfig):
    p, i = default_cfg.process_noise, default_cfg.init_noise
    assert np.allclose(np.diag(default_cfg.Q()), [p.pos**2, p.pos**2, p.heading**2, p.speed**2])
    assert np.allclose(np.diag(default_cfg.P0()), [i.pos**2, i.pos**2, i.heading**2, i.speed**2])
    assert np.count_nonzero(default_cfg.Q() - np.diag(np.diag(default_cfg.Q()))) == 0


def test_overrides_are_flags_not_code_forks():
    cfg = load_config(overrides={"motion_model": "bicycle", "steps": 50, "process_noise": {"pos": 0.5}})
    assert cfg.motion_model == "bicycle" and cfg.steps == 50
    assert cfg.process_noise.pos == 0.5 and cfg.process_noise.heading == 0.002  # merged, not replaced


def test_config_hash_is_stable_and_sensitive(default_cfg: SimConfig):
    assert default_cfg.config_hash() == load_config().config_hash()
    assert default_cfg.config_hash() != load_config(overrides={"seed": 1}).config_hash()


def test_replace_revalidates(default_cfg: SimConfig):
    with pytest.raises(ConfigError):
        default_cfg.replace(motion_model="segway")


@pytest.mark.parametrize(
    "overrides",
    [
        {"nonsense_key": 1},
        {"process_noise": {"posn": 0.1}},
        {"motion_model": "segway"},
        {"dt": 0},
        {"steps": 0},
        {"steps": "many"},
        {"seg_len_range": [300, 100]},
        {"clients": [{"profile": "no_such_profile"}]},
        {"clients": [{"profile": "high_grade", "route_mix": [0, 0, 0]}]},
        {"clients": [{"profile": "high_grade", "route_mix": [1, 0]}]},
        {"clients": [{"profile": "high_grade", "fault_mode": "gremlins"}]},
        {"profiles": {"high_grade": {"gnss": {"p_drop": 1.5}}}},
        {"profiles": {"high_grade": {"gnss": {"sigma": 0}}}},
        {"mixed_route_mix": [1, 1]},
    ],
)
def test_invalid_configs_are_rejected(overrides):
    with pytest.raises(ConfigError):
        load_config(overrides=overrides)


def test_missing_segment_type_rejected(default_cfg: SimConfig):
    # (``overrides`` deep-merges, so build the incomplete mapping via replace().)
    with pytest.raises(ConfigError):
        default_cfg.replace(segments={"urban": default_cfg.segments["urban"]})


def test_override_merges_into_segment_not_replace():
    cfg = load_config(overrides={"segments": {"urban": {"yaw_wander_std": 0.5}}})
    assert cfg.segments["urban"].yaw_wander_std == 0.5
    assert cfg.segments["urban"].speed_range == (5.0, 14.0)  # untouched
    assert set(cfg.segments) == set(SEGMENT_TYPES)
