"""Tables, statistics and figures from raw cell records (torch-free).

Everything printed is computed from the JSON records written by ``runner``. Missing experiments are reported
as "no runs", never filled in. E7 (CARLA) is not implemented and says so.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from afkn.eval.config import EvalConfig
from afkn.eval.fairness import audit
from afkn.eval.metrics import rounds_to_converge
from afkn.eval.stats import mean_std, paired_wilcoxon

CELL_COLS = ["noise", "hetero", "faulty_frac", "participation", "attack", "attack_frac", "variant"]


def load_records(cells_dir: str | Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(cells_dir).glob("*.json"))]


def records_frame(records: list[dict]) -> pd.DataFrame:
    """One row per (record, experiment tag)."""
    rows = []
    for r in records:
        curve = [(int(a), float(b)) for a, b in r.get("curve", [])]
        for exp in r.get("exps", []):
            rows.append({
                "id": r["id"], "exp": exp, "method": r["method"], "seed": r["cell"]["seed"],
                **{c: r["cell"][c] for c in CELL_COLS},
                "final_rmse": r.get("final_rmse", np.nan), "total_bytes": r.get("total_bytes", 0),
                "rounds_to_conv": rounds_to_converge(curve), "elapsed_s": r.get("elapsed_s", np.nan),
            })
    return pd.DataFrame(rows)


def flatten_rounds(records: list[dict]) -> pd.DataFrame:
    """One row per (record, experiment, round, client) with the fields the fairness audit needs."""
    rows = []
    for r in records:
        for rd in r.get("rounds", []):
            k_acc = sum(1 for c in rd["clients"] if c["accepted"])
            for c in rd["clients"]:
                l2 = c.get("layer2", {})
                for exp in r.get("exps", []):
                    rows.append({
                        "id": r["id"], "exp": exp, "method": r["method"], "seed": r["cell"]["seed"],
                        "attack": r["cell"]["attack"], "attack_frac": r["cell"]["attack_frac"],
                        "variant": r["cell"]["variant"], "round": rd["round"], "client": c["client"],
                        "profile": c["profile"], "accepted": bool(c["accepted"]), "reject_reason": c["reject_reason"],
                        "weight": float(c["weight"]), "attacker": bool(c["attacker"]), "k_accepted": k_acc,
                        "probation": bool(l2.get("probation", False)), "liar": bool(l2.get("liar", False)),
                    })
    return pd.DataFrame(rows)


def agg(df: pd.DataFrame, by: list[str], value: str = "final_rmse") -> pd.DataFrame:
    """Mean, std (over seeds), n and the per-seed values for each group."""
    out = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        g = g.sort_values("seed")
        m, s, n = mean_std(g[value].to_numpy())
        out.append({**dict(zip(by, key, strict=True)), "mean": m, "std": s, "n": n,
                    "per_seed": ";".join(f"{v:.4g}" for v in g[value])})
    return pd.DataFrame(out)


def paired(df: pd.DataFrame, a: str, b: str, by: list[str], value: str = "final_rmse", col: str = "method") -> pd.DataFrame:
    """Paired Wilcoxon of method ``a`` vs ``b`` over seeds within each group of ``by``."""
    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        pa = g[g[col] == a].set_index("seed")[value]
        pb = g[g[col] == b].set_index("seed")[value]
        seeds = sorted(set(pa.index) & set(pb.index))
        if not seeds:
            continue
        st = paired_wilcoxon(pa.loc[seeds].to_numpy(), pb.loc[seeds].to_numpy())
        rows.append({**dict(zip(by, key, strict=True)), "a": a, "b": b, **st})
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no runs_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = [f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _save(df: pd.DataFrame, name: str, out: Path, files: list[Path]) -> None:
    p = out / "tables"
    p.mkdir(parents=True, exist_ok=True)
    df.to_csv(p / f"{name}.csv", index=False)
    (p / f"{name}.md").write_text(_md(df), encoding="utf-8")
    files += [p / f"{name}.csv", p / f"{name}.md"]


def _fig(name: str, out: Path, files: list[Path]) -> Path:
    p = out / "figures"
    p.mkdir(parents=True, exist_ok=True)
    files.append(p / f"{name}.png")
    return p / f"{name}.png"


def generate_report(records: list[dict], out_dir: str | Path, ec: EvalConfig) -> list[Path]:
    """Write every table (CSV + Markdown), figure (PNG) and ``summary.md``. Returns the written files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    df = records_frame(records)
    rd = flatten_rounds(records)
    sections: list[str] = []

    def exp(e: str) -> pd.DataFrame:
        return df[df["exp"] == e] if not df.empty else df

    # ---- E1 heterogeneity
    d = exp("E1")
    t = agg(d, ["noise", "hetero"]) if not d.empty else pd.DataFrame()
    _save(t, "E1_heterogeneity_fedavg", out, files)
    sections.append("## E1 effect of heterogeneity (equal-weight FedKalmanNet, global-test position RMSE [m])\n" + _md(t))

    # ---- E2 does confidence help
    d = exp("E2")
    t = agg(d, ["noise", "hetero", "method"]) if not d.empty else pd.DataFrame()
    _save(t, "E2_methods", out, files)
    s = paired(d, "proposed", "fedavg", ["noise", "hetero"]) if not d.empty else pd.DataFrame()
    _save(s, "E2_stats_proposed_vs_fedavg", out, files)
    sections.append("## E2 methods (RMSE [m], mean/std over seeds)\n" + _md(t) +
                    "\nPaired Wilcoxon, proposed vs fedavg (effect_r > 0: proposed has larger error):\n" + _md(s))

    # ---- E3 faulty clients
    d = exp("E3")
    t = agg(d, ["faulty_frac", "method"]) if not d.empty else pd.DataFrame()
    _save(t, "E3_faulty_clients", out, files)
    s = paired(d, "proposed", "fedavg", ["faulty_frac"]) if not d.empty else pd.DataFrame()
    _save(s, "E3_stats_proposed_vs_fedavg", out, files)
    sections.append("## E3 unreliable (faulty-sensor) clients\n" + _md(t) + "\n" + _md(s))

    # ---- E4 trade-offs
    d = exp("E4")
    if not d.empty:
        d = d.assign(MB=d["total_bytes"] / 1e6)
        t = agg(d, ["participation", "method"])
        mb = agg(d, ["participation", "method"], "MB")[["participation", "method", "mean"]].rename(columns={"mean": "MB_mean"})
        rc = agg(d.dropna(subset=["rounds_to_conv"]), ["participation", "method"], "rounds_to_conv")[
            ["participation", "method", "mean"]].rename(columns={"mean": "rounds_to_conv_mean"}) \
            if d["rounds_to_conv"].notna().any() else pd.DataFrame(columns=["participation", "method", "rounds_to_conv_mean"])
        t = t.merge(mb, on=["participation", "method"]).merge(rc, on=["participation", "method"], how="left")
    else:
        t = pd.DataFrame()
    _save(t, "E4_tradeoffs", out, files)
    sections.append("## E4 accuracy vs communication (MB = parameters x 4 bytes x participating clients, both directions)\n" + _md(t))

    # ---- E5 ablations
    d = exp("E5")
    t = agg(d, ["variant", "attack"]) if not d.empty else pd.DataFrame()
    _save(t, "E5_ablations", out, files)
    stats5 = []
    if not d.empty:
        for atk, g in d.groupby("attack"):
            for v in sorted(set(g["variant"]) - {"default"}):
                sub = g[g["variant"].isin(["default", v])]
                st = paired(sub, v, "default", ["attack"], col="variant")
                stats5.append(st)
    s = pd.concat(stats5, ignore_index=True) if stats5 else pd.DataFrame()
    _save(s, "E5_stats_vs_default", out, files)
    sections.append("## E5 ablations (a = variant, b = default; the softmax-temperature ablation does not apply: the "
                    "combiner is a product/mean, not a softmax)\n" + _md(t) + "\n" + _md(s))

    # ---- E6 fairness audit
    fair_rows = []
    prof_rows = []
    if not rd.empty:
        r6 = rd[rd["exp"] == "E6"]
        for m, g in r6.groupby("method"):
            a = audit(g, ec.fairness)
            fair_rows.append({"method": m, "ratio_low_vs_reference": a["ratio_low_vs_reference"], "concern": a["concern"],
                              **{f"rel_weight[{p}]": w for p, w in a["by_profile"].items()}})
    t6 = pd.DataFrame(fair_rows)
    for r in records:
        if "E6" in r.get("exps", []) and "client_test_rmse" in r:
            for prof, v in zip(r["profiles"], r["client_test_rmse"], strict=True):
                prof_rows.append({"method": r["method"], "seed": r["cell"]["seed"], "profile": prof, "client_test_rmse": v})
    e6err = agg(pd.DataFrame(prof_rows), ["method", "profile"], "client_test_rmse") if prof_rows else pd.DataFrame()
    _save(t6, "E6_fairness_weights", out, files)
    _save(e6err, "E6_error_by_client_class", out, files)
    sections.append("## E6 fairness audit (relative weight 1.0 = equal share; `concern` = honest low-grade clients get less than "
                    f"{ec.fairness.min_relative_weight_ratio} of the reference profiles' weight; a flag, not a verdict)\n"
                    + _md(t6) + "\nError of the final global model on each client's own test split, by client class:\n" + _md(e6err))

    # ---- E7
    sections.append("## E7 simulator transfer (CARLA)\n**Not run.** E7 needs the CARLA simulator, which is not available in this "
                    "environment and was not implemented. No number is reported for it.\n")

    # ---- E8 attacks
    d = exp("E8")
    t8, deg, stats8 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if not d.empty:
        t8 = agg(d, ["method", "attack", "attack_frac"])
        clean = d[d["attack"] == "none"].set_index(["method", "seed"])["final_rmse"]
        rows = []
        for _, r in d[d["attack"] != "none"].iterrows():
            if (r["method"], r["seed"]) in clean.index:
                rows.append({**r.to_dict(), "degradation": r["final_rmse"] - clean.loc[(r["method"], r["seed"])]})
        if rows:
            deg = agg(pd.DataFrame(rows), ["method", "attack", "attack_frac"], "degradation")
        sub = d[d["attack"] != "none"]
        if not sub.empty:
            stats8 = pd.concat([paired(sub, "proposed", "fedavg", ["attack", "attack_frac"]),
                                paired(sub, "proposed", "proposed_reported", ["attack", "attack_frac"])], ignore_index=True)
    w8 = pd.DataFrame()
    if not rd.empty:
        r8 = rd[(rd["exp"] == "E8") & rd["accepted"]].copy()
        if not r8.empty:
            r8["rel"] = r8["weight"] * r8["k_accepted"]
            rows = []
            for (m, a, f), g in r8[r8["attack"] != "none"].groupby(["method", "attack", "attack_frac"]):
                low = g[(~g["attacker"]) & g["profile"].isin(ec.fairness.low_profiles)]["rel"]
                rows.append({"method": m, "attack": a, "attack_frac": f,
                             "attacker_rel_weight": g[g["attacker"]]["rel"].mean(),
                             "honest_rel_weight": g[~g["attacker"]]["rel"].mean(),
                             "honest_low_grade_rel_weight": low.mean() if len(low) else np.nan})
            w8 = pd.DataFrame(rows)
    _save(t8, "E8_rmse", out, files)
    _save(deg, "E8_degradation_vs_no_attack", out, files)
    _save(stats8, "E8_stats", out, files)
    _save(w8, "E8_attacker_weights", out, files)
    sections.append("## E8 attacks (compromised clients hold valid keys: Layer 1 passes by design; simulation only, no security guarantee)\n"
                    + _md(t8) + "\nDegradation vs the same method's no-attack run (paired by seed):\n" + _md(deg)
                    + "\nWeights (relative; 1.0 = equal share). Compare `honest_low_grade_rel_weight` with E6:\n" + _md(w8)
                    + "\nPaired Wilcoxon (a vs b):\n" + _md(stats8))

    _figures(df, records, rd, t8, deg, w8, out, files)

    n_runs = len(records)
    header = (f"# Results summary (scale: {ec.scale_name})\n\nGenerated from {n_runs} raw run records by "
              "`afkn.eval.report`. Every number is computed from those runs; nothing is quoted from papers. Starting-default "
              "thresholds, untuned. Failures and null results are reported as they are (equal-weight FedKalmanNet winning is a "
              "valid outcome). Wilcoxon p-values are descriptive when the number of seeds is below 6.\n\n")
    (out / "summary.md").write_text(header + "\n".join(sections), encoding="utf-8")
    files.append(out / "summary.md")
    return files


def _figures(df, records, rd, t8, deg, w8, out: Path, files: list[Path]) -> None:
    # convergence (E2, default fleet)
    curves: dict[str, list[list[tuple[int, float]]]] = {}
    for r in records:
        c = r["cell"]
        if "E2" in r.get("exps", []) and r.get("curve") and c["noise"] == 1.0 and c["hetero"] == "strong":
            curves.setdefault(r["method"], []).append([(int(a), float(b)) for a, b in r["curve"]])
    if curves:
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, cs in sorted(curves.items()):
            n = min(len(c) for c in cs)
            xs = [p[0] for p in cs[0][:n]]
            ys = np.mean([[p[1] for p in c[:n]] for c in cs], axis=0)
            ax.plot(xs, ys, marker="o", label=m)
        ax.set_xlabel("round")
        ax.set_ylabel("global-test position RMSE [m]")
        ax.set_title("E2 convergence (mean over seeds)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(_fig("E2_convergence", out, files), dpi=150)
        plt.close(fig)
    # E3
    d = df[df["exp"] == "E3"] if not df.empty else df
    if not d.empty:
        t = agg(d, ["faulty_frac", "method"])
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, g in t.groupby("method"):
            ax.errorbar(g["faulty_frac"], g["mean"], yerr=g["std"], marker="o", capsize=3, label=m)
        ax.set_xlabel("fraction of faulty clients")
        ax.set_ylabel("RMSE [m]")
        ax.set_title("E3 unreliable clients")
        ax.legend()
        fig.tight_layout()
        fig.savefig(_fig("E3_faulty_clients", out, files), dpi=150)
        plt.close(fig)
    # E4
    d = df[df["exp"] == "E4"] if not df.empty else df
    if not d.empty:
        d = d.assign(MB=d["total_bytes"] / 1e6)
        g = d.groupby(["method", "participation"]).agg(MB=("MB", "mean"), rmse=("final_rmse", "mean")).reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, gg in g.groupby("method"):
            ax.plot(gg["MB"], gg["rmse"], marker="o", label=m)
            for _, r in gg.iterrows():
                ax.annotate(f"{r['participation']:.1f}", (r["MB"], r["rmse"]), fontsize=7)
        ax.set_xlabel("MB transmitted (up + down)")
        ax.set_ylabel("RMSE [m]")
        ax.set_title("E4 accuracy vs communication (labels: participation)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(_fig("E4_accuracy_vs_mb", out, files), dpi=150)
        plt.close(fig)
    # E5
    d = df[df["exp"] == "E5"] if not df.empty else df
    if not d.empty:
        t = agg(d, ["variant", "attack"])
        variants = list(dict.fromkeys(t["variant"]))
        attacks = list(dict.fromkeys(t["attack"]))
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.8 / max(len(attacks), 1)
        for j, a in enumerate(attacks):
            vals = [t[(t["variant"] == v) & (t["attack"] == a)]["mean"].mean() for v in variants]
            ax.bar(np.arange(len(variants)) + j * w, vals, w, label=f"attack={a}")
        ax.set_xticks(np.arange(len(variants)) + 0.4 - w / 2)
        ax.set_xticklabels(variants, rotation=30, ha="right")
        ax.set_ylabel("RMSE [m]")
        ax.set_title("E5 ablations")
        ax.legend()
        fig.tight_layout()
        fig.savefig(_fig("E5_ablations", out, files), dpi=150)
        plt.close(fig)
    # E6
    if not rd.empty:
        r6 = rd[(rd["exp"] == "E6") & rd["accepted"]].copy()
        if not r6.empty:
            r6["rel"] = r6["weight"] * r6["k_accepted"]
            g = r6.groupby(["method", "profile"])["rel"].mean().unstack("method")
            ax = g.plot(kind="bar", figsize=(7, 4))
            ax.axhline(1.0, color="k", lw=0.8, ls="--")
            ax.set_ylabel("mean relative weight (1 = equal share)")
            ax.set_title("E6 weights by client class (honest clients)")
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(_fig("E6_weights_by_profile", out, files), dpi=150)
            plt.close(fig)
    # E8
    if not deg.empty:
        attacks = sorted(set(deg["attack"]))
        fig, axes = plt.subplots(1, len(attacks), figsize=(4 * len(attacks), 4), squeeze=False)
        for ax, a in zip(axes[0], attacks, strict=True):
            for m, g in deg[deg["attack"] == a].groupby("method"):
                g = g.sort_values("attack_frac")
                ax.errorbar(g["attack_frac"], g["mean"], yerr=g["std"], marker="o", capsize=3, label=m)
            ax.set_title(a)
            ax.set_xlabel("malicious fraction")
        axes[0][0].set_ylabel("RMSE degradation vs no attack [m]")
        axes[0][-1].legend()
        fig.tight_layout()
        fig.savefig(_fig("E8_degradation", out, files), dpi=150)
        plt.close(fig)
