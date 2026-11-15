# FedKal-AV: Confidence-Aware Federated KalmanNet for AV Localization

[![CI](https://github.com/Krushnakant-08/FedKal-AV/actions/workflows/ci.yml/badge.svg)](https://github.com/Krushnakant-08/FedKal-AV/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Honors-with-Research project (A.Y. 2026-27). A simulation study of federated, neural-aided Kalman
filtering for vehicle localization under non-IID sensor noise, GNSS outages and faulty or malicious
clients, with an identity layer (Ed25519) and a server-verified confidence layer.

> **Status: work in progress. No experiments have been run and there are no results yet.**
> All default thresholds and noise levels are starting values to tune, not findings.
> This is a simulation study; it makes no formal privacy or security claims.

## Build status

| Step | Component | Package | Status |
| :-- | :-- | :-- | :-- |
| 1 | Motion model, GNSS/IMU noise, 5 client profiles, trusted set | `afkn/sim`, `afkn/data` | **in PR** (`step-1-sim`) |
| 2 | EKF and UKF baselines | `afkn/filters` | not started |
| 3 | KalmanNet-style GRU gain network | `afkn/models` | not started |
| 4 | Client, server, FedAvg loop, byte-level messages | `afkn/fed` | not started |
| 5 | Ed25519 signing and the identity checks | `afkn/security` | not started |
| 6 | Layer 2 scores and confidence aggregator | `afkn/confidence` | not started |
| 7 | Attacks | `afkn/attacks` | not started |
| 8 | Experiments E1-E8, statistics, fairness audit | `afkn/eval` | not started |

## Setup

```bash
git clone https://github.com/Krushnakant-08/FedKal-AV.git
cd FedKal-AV
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.11+. CPU is sufficient. (Step 1 itself needs only NumPy, SciPy and PyYAML; PyTorch is
used from Step 3 on.)

## Run the tests

```bash
pytest -q
```

## Inspect the generated dataset

Builds the full dataset in memory (about 4 s, about 0.35 GB RAM) and prints realised statistics.
Writes nothing to disk.

```bash
python -m afkn.scripts.inspect_dataset              # default config, cfg.seed
python -m afkn.scripts.inspect_dataset --seed 7 --json
```

## Step 1 contract (what later steps rely on)

State `x = [px, py, heading, speed]`, input `u = [yaw_rate, accel]`, `dt = 0.1 s`, 1000 steps.

```
x[t]  true state,  t = 0..T      (N, T+1, 4)    heading is NOT wrapped; use motion.wrap_angle on differences
u[t]  drives x[t] -> x[t+1]      (N, T, 2)      u_true (ground truth) and u_imu (what filters may use)
z[t]  GNSS measurement OF x[t+1] (N, T, 2)      NaN exactly where z_mask is False
```

* `afkn.seeding.SeedTree`: every random stream is a pure function of `(seed, named path)`.
  Same seed gives bit-identical data (`TrajectoryBatch.content_hash()`).
* `afkn.sim.config`: frozen, validated dataclasses loaded from YAML (`afkn/configs/default.yaml`).
  Ablations are config overrides, e.g. `load_config(overrides={"motion_model": "bicycle"})`.
* `afkn.sim.motion`: `f`, `jac_x`, `jac_u`, `h`, `H` shared by the simulator, the filters and
  KalmanNet. Models: `ctrv` (exact arc) and `bicycle` (midpoint-heading Euler).
* `afkn.sim.sensors`: `DeviceParams` (the *realised* per-client sensor constants; `R()` gives the
  honest GNSS covariance), `apply_imu`, `apply_gnss`.
* `afkn.data.build_federated_dataset(cfg, seed)` returns per-client train/val/test, a shared
  global test set and a server trusted set. Both mixed sets cover all five profiles and are
  disjoint from every client.
* `cfg.Q()` and `cfg.P0()` give the process-noise and initial covariances used by the simulator.

## Layout

```text
afkn/
  seeding.py        SeedTree (SeedSequence-based, order-independent)
  sim/              config, motion, sensors, generator, batch (TrajectoryBatch)
  data/             federated partition
  configs/          default.yaml
  scripts/          inspect_dataset.py   (make_all.py arrives in Step 8)
tests/              one test module per component; pytest -q runs everything
STATE.md            development log
THIRD_PARTY.md      dependency licenses
```

## License

MIT, see [LICENSE](LICENSE).
