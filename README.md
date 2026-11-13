# FedKal-AV: Federated KalmanNet for Autonomous Vehicle Navigation

[![CI](https://github.com/Krushnakant-08/FedKal-AV/actions/workflows/ci.yml/badge.svg)](https://github.com/Krushnakant-08/FedKal-AV/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

FedKal-AV is a federated learning framework for neural Kalman filtering (KalmanNet) tailored for autonomous vehicle state estimation under non-IID sensor noise, intermittent GNSS denial, and sensor faults.

## Features

- **High-Fidelity Vehicle Simulation**: Constant Turn Rate and Velocity (CTRV) and kinematic bicycle motion models with configurable process noise.
- **Realistic Sensor Modeling**:
  - **GNSS**: Position measurements with additive Gaussian noise, calibration bias, random dropout, and scheduled outage windows (e.g., tunnels).
  - **IMU**: Yaw-rate and longitudinal acceleration with constant bias, random-walk bias drift, scale error, and high-frequency noise.
- **Heterogeneous Non-IID Fleet**: 5 canonical client profiles (high-grade, mid-grade, low-grade, intermittent GNSS-denied, faulty) capturing realistic autonomous vehicle fleet diversity.
- **Rigorous Dataset Partitioning**: Per-client train/val/test splits along with shared global test and server-trusted verification sets.
- **Federated KalmanNet Architecture**: Robust state estimation combining physics-based kinematic prediction with deep learning-based Kalman gain adaptation.

## Installation

Ensure Python 3.11 or later is installed:

```bash
git clone https://github.com/Krushnakant-08/FedKal-AV.git
cd FedKal-AV
pip install -r requirements.txt
pip install -e .
```

## Running Tests

Run the test suite with pytest:

```bash
pytest tests/ -v --tb=short
```

## Architecture Overview

```text
FedKal-AV/
├── simulation/            # Vehicle dynamics, sensors, trajectory & dataset generation
│   ├── config.py          # Dataclasses for motion, sensors, clients, and datasets
│   ├── motion.py          # CTRV and Kinematic Bicycle motion propagators
│   ├── trajectory.py      # Multi-segment trajectory generator (urban, highway, tunnel)
│   └── dataset.py         # Federated non-IID dataset builder
├── tests/                 # Unit and integration test suite
│   └── test_simulation.py # Comprehensive verification of simulation and models
├── .github/workflows/ci.yml # Continuous Integration workflow
├── pyproject.toml         # Build system and package configuration
├── requirements.txt       # Project dependencies
└── STATE.md               # Project development roadmap and progress tracker
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
