"""Fairness audit (torch-free): are honest low-grade clients treated like attackers?

Input rows are per-client per-round log rows (see ``report.flatten_rounds``): ``profile``, ``weight``, ``accepted``,
``attacker``, ``k_accepted``, ``method``. The comparable quantity is the RELATIVE weight ``weight x k_accepted``
(1.0 = an equal share), which does not depend on how many clients took part in the round.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from afkn.eval.config import FairnessSpec


def relative_weight(rows: pd.DataFrame) -> pd.Series:
    return rows["weight"] * rows["k_accepted"]


def audit(rows: pd.DataFrame, spec: FairnessSpec) -> dict:
    """Mean relative weight per profile, low-grade vs reference ratio, and (if present) attacker weight.

    Only accepted, non-attacking clients count as "honest". ``concern`` is True when the honest low-grade clients
    get less than ``min_relative_weight_ratio`` of the reference profiles' relative weight. It is a flag for the
    reader, not a pass/fail claim.
    """
    if rows.empty:
        return {"by_profile": {}, "ratio_low_vs_reference": float("nan"), "concern": False, "attacker_relative_weight": float("nan")}
    acc = rows[rows["accepted"]].copy()
    acc["rel"] = relative_weight(acc)
    honest = acc[~acc["attacker"]]
    by_profile = {p: float(g["rel"].mean()) for p, g in honest.groupby("profile")}
    low = honest[honest["profile"].isin(spec.low_profiles)]["rel"]
    ref = honest[honest["profile"].isin(spec.reference_profiles)]["rel"]
    ratio = float(low.mean() / ref.mean()) if len(low) and len(ref) and ref.mean() > 0 else float("nan")
    att = acc[acc["attacker"]]["rel"]
    return {
        "by_profile": by_profile,
        "ratio_low_vs_reference": ratio,
        "concern": bool(np.isfinite(ratio) and ratio < spec.min_relative_weight_ratio),
        "attacker_relative_weight": float(att.mean()) if len(att) else float("nan"),
    }
