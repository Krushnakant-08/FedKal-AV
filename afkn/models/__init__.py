"""KalmanNet-style learned Kalman gain.

Only the torch-free configuration is re-exported here so ``import afkn.models`` works without
PyTorch. Import the network and trainer explicitly::

    from afkn.models.kalmannet import KalmanNet
    from afkn.models.train import train, evaluate
"""

from afkn.models.config import (
    N_FEATURES,
    N_GAIN,
    KalmanNetConfig,
    KalmanNetSpec,
    TrainSpec,
    expected_num_params,
    load_kalmannet_config,
)

__all__ = [
    "N_FEATURES", "N_GAIN", "KalmanNetConfig", "KalmanNetSpec", "TrainSpec",
    "expected_num_params", "load_kalmannet_config",
]
