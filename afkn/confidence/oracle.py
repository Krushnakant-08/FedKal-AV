"""What the server can compute for itself on its trusted set (clients cannot forge it)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class TrustedOracle(Protocol):
    """Server-side measurements on the trusted set."""

    def reference_delta(self, global_params: np.ndarray, round_idx: int) -> np.ndarray:
        """Reference update Delta_0: a training step from ``global_params`` on the trusted set."""
        ...

    def rmse(self, params: np.ndarray) -> float:
        """Position RMSE of the model ``params`` on the trusted set (may be NaN/inf for broken models)."""
        ...
