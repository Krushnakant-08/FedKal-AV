"""Confidence-aware aggregation (Layer 2). Runs only on updates that passed Layer 1.

Per round, for the accepted updates:

1. norm clip to ``clip_c`` x median norm;
2. ``s_dir = max(0, cos(delta_i, delta_0))`` against the server's reference update on its trusted set;
3. ``s_perf = sigmoid(gain / perf_scale)``, ``gain = RMSE(global) - RMSE(global + delta_i)`` measured by the server;
4. claim check: reported val-RMSE gain vs measured gain -> "liar" flag, score x ``liar_factor``;
5. ``score = s_dir x s_perf`` (or mean), then ``c_i = EMA(score)``;
6. three consecutive rounds with score < ``low_score`` -> probation for ``probation_len`` rounds (the trigger
   round included): raw weight 0 and the update is logged as "probation", never silently dropped. The floor
   still applies, so such a client gets exactly the floor weight (deviation from the Build Plan's "weight 0",
   made because the project rules say the floor is always enforced). On leaving
   probation the EMA is reset so the first re-test score defines it;
7. ``raw_i = n_i x c_i``; the loop normalises with the floor (``floor + (1 - k floor) raw / sum raw``).

ASSUMPTIONS (flagged in the PR): the way ``s_dir`` and ``s_perf`` are combined is not specified by the plan
(product by default; ``combine: mean`` is the alternative); an unverifiable (missing/NaN) claim counts as a lie.
"""

from __future__ import annotations

import numpy as np

from afkn.confidence.config import ConfidenceSpec
from afkn.confidence.oracle import TrustedOracle
from afkn.confidence.scores import claim_flags, clip_to_median, combine_scores, direction_scores, perf_scores
from afkn.fed.aggregate import RoundContext, Weighting
from afkn.fed.messages import ClientUpdate


class ConfidenceAggregator:
    """Stateful weighting policy: pass it as ``weighter=`` to ``run_federated``."""

    def __init__(self, spec: ConfidenceSpec, oracle: TrustedOracle) -> None:
        self.spec = spec
        self.oracle = oracle
        self.ema: dict[int, float | None] = {}
        self.streak: dict[int, int] = {}
        self.probation_until: dict[int, int] = {}  # last round (inclusive) spent in probation

    def on_probation(self, client_id: int, round_idx: int) -> bool:
        return round_idx <= self.probation_until.get(client_id, -1)

    def __call__(self, updates: list[ClientUpdate], ctx: RoundContext) -> Weighting:
        sp = self.spec
        k = len(updates)
        ids = [u.client_id for u in updates]
        deltas = np.stack([np.asarray(u.delta, dtype=np.float64) for u in updates])
        n = np.array([u.n_samples for u in updates], dtype=np.float64)
        norms = np.linalg.norm(deltas, axis=1)

        if sp.use_clip:
            clipped, factor, med = clip_to_median(deltas, sp.clip_c)
        else:
            clipped, factor, med = deltas, np.ones(k), float(np.median(norms))

        g = np.asarray(ctx.global_params, dtype=np.float64)
        s_dir = np.ones(k)
        if sp.use_direction:
            ref = np.asarray(self.oracle.reference_delta(ctx.global_params, ctx.round), dtype=np.float64)
            s_dir = direction_scores(clipped, ref)

        reported = np.array([
            u.signals.get("val_rmse_before", np.nan) - u.signals.get("val_rmse_after", np.nan) for u in updates
        ], dtype=np.float64)
        gain = np.full(k, np.nan)
        s_perf = np.ones(k)
        if (sp.use_perf and sp.perf_source == "server") or sp.use_claim:
            base = float(self.oracle.rmse(ctx.global_params))
            cand = np.array([float(self.oracle.rmse((g + c).astype(np.float32))) for c in clipped])
            gain = base - cand
            if sp.relative_gain and np.isfinite(base) and base > 0:
                gain = gain / base
        if sp.use_perf:
            s_perf = perf_scores(reported if sp.perf_source == "reported" else gain, sp.perf_scale)

        liar = np.zeros(k, dtype=bool)
        if sp.use_claim:
            liar = claim_flags(reported, gain, sp.claim_reported_min, sp.claim_measured_max)

        score = combine_scores(s_dir, s_perf, sp.combine) * np.where(liar, sp.liar_factor, 1.0)
        score = np.clip(np.nan_to_num(score, nan=0.0), 0.0, 1.0)

        raw = np.zeros(k)
        info: dict[int, dict] = {}
        for i, cid in enumerate(ids):
            probation_now = self.on_probation(cid, ctx.round)
            started = False
            ended = False
            if probation_now:
                ema = self.ema.get(cid)
            else:
                if self.probation_until.get(cid, -1) == ctx.round - 1:
                    ended = True  # re-test round: forget the pre-probation history
                    self.ema[cid] = None
                    self.streak[cid] = 0
                prev = self.ema.get(cid)
                ema = float(score[i]) if (prev is None or not sp.use_history) else (1 - sp.ema_alpha) * prev + sp.ema_alpha * float(score[i])
                self.ema[cid] = ema
                self.streak[cid] = self.streak.get(cid, 0) + 1 if score[i] < sp.low_score else 0
                if self.streak[cid] >= sp.low_rounds:
                    started = True
                    self.probation_until[cid] = ctx.round + sp.probation_len - 1
                    self.streak[cid] = 0
                    self.ema[cid] = None
            in_probation = probation_now or started
            raw[i] = 0.0 if in_probation else float(n[i] * (ema if ema is not None else 0.0))
            info[cid] = {
                "clip_factor": float(factor[i]), "norm": float(norms[i]), "median_norm": med,
                "s_dir": float(s_dir[i]), "s_perf": float(s_perf[i]), "gain": float(gain[i]),
                "reported_gain": float(reported[i]), "liar": bool(liar[i]), "score": float(score[i]),
                "ema": None if ema is None else float(ema), "streak": self.streak.get(cid, 0),
                "probation": bool(in_probation), "probation_started": started, "probation_ended": ended,
                "reason": "probation" if in_probation else None,
            }
        return Weighting(raw=raw, info=info, deltas=clipped.astype(np.float32), floor=sp.weight_floor)
