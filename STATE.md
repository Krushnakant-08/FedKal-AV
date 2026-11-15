# FedKal-AV Development State

Source of truth for step numbering: `Vehicle_Verification_and_Build_Plan.docx` (project knowledge).
Nothing has been run experimentally. All thresholds and noise levels are untuned starting defaults.

## Progress

| Step | Component | Package | Status |
| :-- | :-- | :-- | :-- |
| 0 | Project skeleton and CI | repo root | merged |
| 1 | Motion model, GNSS/IMU noise, 5 client profiles, trusted set | `afkn/sim`, `afkn/data` | **merged** (PR #1) |
| 2 | EKF and UKF baselines | `afkn/filters` | next |
| 3 | KalmanNet-style GRU gain network | `afkn/models` | not started |
| 4 | Client, server, FedAvg loop, byte-level messages | `afkn/fed` | not started |
| 5 | Ed25519 signing and the identity checks | `afkn/security` | not started |
| 6 | Layer 2 scores and confidence aggregator | `afkn/confidence` | not started |
| 7 | Attacks | `afkn/attacks` | not started |
| 8 | Experiments E1-E8, statistics, fairness audit | `afkn/eval` | not started |

Correction: an earlier version of this file marked Step 1 "Completed" under a `simulation/` package
and a different step table. That code never existed on `main`; Step 1 was built from scratch in PR #1.

## Step 1: files and interfaces

- `afkn/seeding.py`: `SeedTree(entropy).child(*keys).rng()/int_seed()`; every stream is a pure function of (seed, named path).
- `afkn/sim/config.py`: frozen validated dataclasses, `load_config(path=None, overrides=None)`, `SimConfig.Q()`, `.P0()`, `.profile_names`, `.config_hash()`. Defaults in `afkn/configs/default.yaml`.
- `afkn/sim/motion.py`: `f`, `jac_x`, `jac_u`, `h`, `H`, `wrap_angle`, `steering_angle`; models `ctrv` (exact arc) and `bicycle` (midpoint Euler).
- `afkn/sim/sensors.py`: `DeviceParams` (realised per-client constants, `.R()`), `realize_device`, `apply_imu`, `apply_gnss`.
- `afkn/sim/generator.py`: `plan_routes`, `plan_controls`, `rollout_states`, `generate_batch`.
- `afkn/sim/batch.py`: `TrajectoryBatch` (`x`, `u_true`, `u_imu`, `z`, `z_mask`, `seg`, `x0_est`, diagnostics, `profile_idx`), `content_hash()`, `row_hashes()`, `subset`, `concat`.
- `afkn/data/partition.py`: `build_federated_dataset(cfg, seed)`, `build_client_data`, `build_mixed_set`; `FederatedDataset.fingerprint()/summary()`.
- `afkn/scripts/inspect_dataset.py`: prints realised dataset statistics.

Time convention: `u[t]` drives `x[t] -> x[t+1]`; `z[t]` measures `x[t+1]`; heading is not wrapped.
Filters may use only `u_imu`, `z`, `z_mask`, `x0_est`.

## Open issues / notes for later steps

- `requirements.txt` is not pinned (project rule asks for pinned); pin once versions are verified on Python 3.11, before Step 3.
- Process noise Q is additive on the state and excludes IMU error: the Step 2 NEES test must use an ideal IMU or an effective Q built from `jac_u`.
- `faulty` clients are honest-but-faulty sensors, not attackers; the trusted/global-test `faulty` slice uses the profile default fault mode (`constant_bias`).
- Full default dataset is about 0.35 GB of float64 in memory.
- Tests ran locally on Python 3.13; CI uses 3.11.
