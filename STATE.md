# FedKal-AV Development State

This document tracks the incremental progress and implementation status across development steps for the FedKal-AV project.

## Progress Overview

| Step | Component | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Step 0** | **Project Skeleton & CI** | **Completed** | Repository initialized, metadata, packaging, CI workflow, and licensing set up. |
| **Step 1** | **Simulation Module & Synthetic Dataset** | **Completed** | CTRV and kinematic bicycle motion models, multi-segment trajectories, GNSS/IMU sensor modeling, 5 non-IID client profiles, deterministic dataset partitioning. |
| **Step 2** | **Centralized KalmanNet Baseline** | Pending | PyTorch KalmanNet architecture, GRU/RNN gain predictor, supervised training loop, evaluation on synthetic dataset. |
| **Step 3** | **Federated Learning Infrastructure** | Pending | Federated averaging (FedAvg), client-server communication abstraction, local training pipelines. |
| **Step 4** | **Robust Aggregation & Defenses** | Pending | Byzantine fault tolerance, trimmed mean, coordinate-wise median, detection of faulty clients (e.g., bias injection). |
| **Step 5** | **Intermittent GNSS & Edge Handling** | Pending | Masked loss training, extrapolation during GNSS denial, dynamic adaptation. |
| **Step 6** | **Comprehensive Benchmarking & Analysis** | Pending | Full evaluation comparing EKF, Centralized KalmanNet, FedAvg KalmanNet, and Robust FedKal-AV across noise regimes. |

---

## Detailed Step Log

### Step 0: Project Skeleton & CI
- Initialized repository structure.
- Configured `pyproject.toml` and `requirements.txt`.
- Set up GitHub Actions CI workflow with Python 3.11, pip caching, and automated pytest execution.
- Configured `.gitignore`, `LICENSE` (MIT, 2024), and `THIRD_PARTY.md`.

### Step 1: Simulation Module & Synthetic Dataset
- **Configuration (`simulation/config.py`)**: Dataclasses for `MotionConfig`, `GNSSConfig`, `IMUConfig`, `ClientConfig`, and `DatasetConfig`.
- **Vehicle Dynamics (`simulation/motion.py`)**: Exact CTRV propagator with singular yaw-rate limit handling and Kinematic Bicycle model.
- **Trajectory Generation (`simulation/trajectory.py`)**: Multi-segment routing (`urban`, `highway`, `tunnel`), GNSS dropouts, outage windows, and IMU sensor modeling (bias, random-walk drift, scale factor).
- **Federated Dataset Builder (`simulation/dataset.py`)**: 10 clients across 5 profiles (High-grade, Mid-grade, Low-grade, Intermittent, Faulty), train/val/test splits, global test set, and server-trusted verification set.
- **Verification (`tests/test_simulation.py`)**: Unit and statistical tests validating determinism, empirical noise matching configured sigma, dropout rates, and non-IID partition integrity.
