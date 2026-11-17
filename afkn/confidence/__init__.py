"""Layer 2: is this update trustworthy? Scores, history and the confidence aggregator.

Torch-free parts are exported here; ``afkn.confidence.oracle_torch.TorchOracle`` needs PyTorch.
"""

from afkn.confidence.aggregator import ConfidenceAggregator
from afkn.confidence.config import ConfidenceSpec, load_confidence_config
from afkn.confidence.oracle import TrustedOracle
from afkn.confidence.scores import (
    claim_flags,
    clip_to_median,
    combine_scores,
    direction_scores,
    perf_scores,
)

__all__ = [
    "ConfidenceAggregator", "ConfidenceSpec", "TrustedOracle", "claim_flags", "clip_to_median",
    "combine_scores", "direction_scores", "load_confidence_config", "perf_scores",
]
