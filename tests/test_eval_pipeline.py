"""End-to-end smoke test of ``make_all`` (needs PyTorch; in CI it must run, locally it skips without torch).

Done when (Step 8): one script regenerates all tables and figures. This runs the tiny `smoke` scale through the real
runner, so it only proves the pipeline works; it says nothing about the research questions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

if os.environ.get("CI"):
    import torch  # noqa: F401  hard import: CI must never skip this silently
else:
    pytest.importorskip("torch")

from afkn.scripts.make_all import main


@pytest.fixture(scope="module")
def smoke_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    runs, res = root / "runs", root / "results"
    rc = main(["--scale", "smoke", "--out", str(runs), "--results", str(res)])
    assert rc == 0
    return runs, res


def test_make_all_writes_tables_figures_and_summary(smoke_dirs) -> None:
    runs, res = smoke_dirs
    for name in ("E1_heterogeneity_fedavg", "E2_methods", "E3_faulty_clients", "E4_tradeoffs", "E5_ablations",
                 "E6_fairness_weights", "E8_rmse", "E8_degradation_vs_no_attack"):
        assert (res / "tables" / f"{name}.csv").stat().st_size > 0, name
    for name in ("E2_convergence", "E3_faulty_clients", "E4_accuracy_vs_mb", "E5_ablations", "E6_weights_by_profile",
                 "E8_degradation"):
        assert (res / "figures" / f"{name}.png").stat().st_size > 0, name
    summary = (res / "summary.md").read_text()
    assert "Not run" in summary and "E8" in summary


def test_raw_records_log_everything_required(smoke_dirs) -> None:
    runs, _ = smoke_dirs
    recs = [json.loads(p.read_text()) for p in (runs / "cells").glob("*.json")]
    assert recs and all("final_rmse" in r and "exps" in r for r in recs)
    prop = [r for r in recs if r["method"] == "proposed" and r["rounds"]]
    assert prop
    for r in prop[:3]:
        for rd in r["rounds"]:
            assert rd["bytes_down"] > 0 and rd["bytes_up"] > 0
            for c in rd["clients"]:
                assert {"accepted", "reject_reason", "weight", "bytes_up", "signals", "attacker", "profile"} <= set(c)
                if c["accepted"]:
                    assert {"s_dir", "s_perf", "ema", "liar", "score", "probation"} <= set(c["layer2"])
    attacked = [r for r in recs if r["cell"]["attack"] != "none" and r["rounds"]]
    assert attacked and all(r["attackers"] for r in attacked)


def test_rerun_resumes_without_recomputing(smoke_dirs, capsys) -> None:
    runs, res = smoke_dirs
    before = {p.name: p.stat().st_mtime_ns for p in (runs / "cells").glob("*.json")}
    assert main(["--scale", "smoke", "--out", str(runs), "--results", str(res)]) == 0
    after = {p.name: p.stat().st_mtime_ns for p in (runs / "cells").glob("*.json")}
    assert before == after
    assert Path(res / "summary.md").exists()


def test_report_only_and_unknown_experiment(smoke_dirs) -> None:
    runs, res = smoke_dirs
    assert main(["--scale", "smoke", "--out", str(runs), "--results", str(res / "again"), "--report-only"]) == 0
    assert main(["--scale", "smoke", "--experiments", "E7", "--report-only"]) == 2
