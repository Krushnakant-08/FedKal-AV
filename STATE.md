# FedKal-AV Development State

Source of truth for step numbering: `Vehicle_Verification_and_Build_Plan.docx` (project knowledge).
Nothing has been run experimentally. All thresholds and noise levels are untuned starting defaults.

## Progress

| Step | Component | Package | Status |
| :-- | :-- | :-- | :-- |
| 0 | Project skeleton and CI | repo root | merged |
| 1 | Motion model, GNSS/IMU noise, 5 client profiles, trusted set | `afkn/sim`, `afkn/data` | **merged** (PR #1) |
| 2 | EKF and UKF baselines (+ simplified ESKF) | `afkn/filters` | **merged** (PR #3) |
| 3 | KalmanNet-style GRU gain network | `afkn/models` | **merged** (PR #5; torch code written without a local torch, first run in CI) |
| 4 | Client, server, FedAvg loop, byte-level messages | `afkn/fed` | **merged** (PR #6) |
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

## Step 2: files and interfaces

- `afkn/filters/params.py`: `FilterParams(Q, R, Qu, P0, model, dt)`; `matched_params(cfg, device)`, `matched_params_by_profile(cfg, profile_idx)` (per-trajectory R/Qu for mixed sets), `.scaled(q, r, qu)` for deliberate mismatch. `Qu` = IMU white-noise covariance; effective process covariance is `Q + G Qu G^T`.
- `afkn/filters/{ekf,ukf,eskf}.py`: `EKF(params)`, `UKF(params, spec)`, `ESKF(params, spec)`, each `.run(batch, P0=None) -> FilterResult`. ESKF is a SIMPLIFIED planar error-state filter with 2 IMU bias states; with `estimate_bias=False` it equals the EKF.
- `afkn/filters/base.py`: `FilterResult` (`x_hat`/`P` aligned with `batch.x` incl. index 0 = `x0_est`; `x_pred`, `P_pred`, `innov`, `S`, `updated`, `nis`, `bias_hat`).
- `afkn/filters/consistency.py`: `nees`, `anees_bounds`, `consistency_summary`, `position_rmse`, `state_error`.
- `afkn/filters/config.py`, `afkn/configs/filters.yaml`: UKF/ESKF settings (`load_filter_config`).
- `afkn/sim/config.py`: added public `build_dataclass`.

Results on the Step 2 test data (matched Q/R, no IMU bias/RW/scale, 200 x 400): EKF ANEES 4.033, NIS 2.007; UKF 4.035 / 2.007 (95% ANEES bounds 3.62-4.40). Exploratory, not experiment results.

## Step 3: files and interfaces

- `afkn/models/config.py` (torch-free): `KalmanNetSpec`, `TrainSpec`, `KalmanNetConfig`, `expected_num_params`, `load_kalmannet_config`; `configs/kalmannet.yaml`.
- `afkn/models/motion_torch.py`: `f_torch`, `h_torch`, `wrap_angle_torch` (parity-tested against `sim.motion`).
- `afkn/models/kalmannet.py`: `KalmanNet(config, model, dt, seed)`; `.run(batch) -> KalmanNetResult` (duck-typed like `FilterResult` for `position_rmse`); `.get_flat_params()/.set_flat_params()` (float32, validated); 30,664 params. Gain is tanh-bounded: |K_ij| < gain_scale_i (added after CI showed divergence with an unbounded gain).
- `afkn/models/train.py`: `train(model, batch, train_cfg, seed) -> TrainLog` (Adam, clip, truncated BPTT; batches with a non-finite loss are skipped and counted in `skipped_batches`), `evaluate`.

## Step 4: files and interfaces

- `afkn/fed/messages.py`: `GlobalBroadcast`, `ClientUpdate` (strict byte encodings; `signing_payload()`, `with_signature()`, `delta_hash`, `payload_bytes`, `wire_bytes`), `MessageError`.
- `afkn/fed/aggregate.py`: `fedavg_weights`, `normalize_weights(raw, floor)` (floor holds exactly), `sample_participants`, `apply_deltas`.
- `afkn/fed/{client,server,loop}.py`: `Client.local_update`, `Server.broadcast/basic_checks/final_weights/aggregate` (`basic_checks` is the hook Step 5 replaces), `run_federated(..., weighter=)` (hook for Step 6), `run_local_only`, `run_centralized`, `run_comparison`.
- Config: `configs/fed.yaml` (`FedSpec`).

## Findings that constrain later steps

- CI exposed that an unbounded KalmanNet gain can destabilise the filter recursion (local-only RMSE ~3e6 m, centralized NaN); fixed with a tanh bound. Federated-vs-local results are still unmeasured beyond the CI test.

- Mismatched Q or R (x0.01 / x100) changes EKF position RMSE by only 2-4% (6.55 m -> 6.68-6.81 m); the damage shows in NEES/NIS. Step 3's "KalmanNet beats the EKF under Q/R mismatch" test needs a regime where mismatch matters (more GNSS-denied time, unmodelled IMU bias, or much stronger mismatch).
- `ctrv` vs `bicycle` differ by at most 3 mm over the whole dataset at dt = 0.1 s; the flag will not drive experimental differences.
- EKF/UKF do not model IMU bias, random walk or scale error (only IMU white noise); matched-noise consistency holds only when those are zero.
- Simplified ESKF with bias states: lower RMSE (about 10-17%) under a large constant IMU bias, about 12% higher RMSE with no bias (single exploratory scenarios).

## Open issues / notes for later steps

- `requirements.txt` is not pinned (project rule asks for pinned); pin once versions are verified on Python 3.11 (needs torch, see below).
- Process noise Q is additive on the state and excludes IMU error; filters model IMU white noise via `Qu` only.
- `faulty` clients are honest-but-faulty sensors, not attackers; the trusted/global-test `faulty` slice uses the profile default fault mode (`constant_bias`).
- Full default dataset is about 0.35 GB of float64 in memory.
- Tests run locally on Python 3.13; CI runs 3.11 and passed on `main` after Steps 1 and 2.
- PyTorch is not installed in the build sandbox and package installs are disabled there, so any torch code cannot be run locally; it can only be run by CI or after the network/package-install setting is enabled (a new chat picks it up).
