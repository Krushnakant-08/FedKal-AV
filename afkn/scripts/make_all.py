"""Regenerate every raw run, table and figure with one command.

    python -m afkn.scripts.make_all --scale smoke            # pipeline check (tiny, minutes on CPU)
    python -m afkn.scripts.make_all --scale small            # laptop sanity run
    python -m afkn.scripts.make_all --scale full             # Build Plan sizes (very slow on CPU)
    python -m afkn.scripts.make_all --scale small --report-only   # rebuild tables/figures from existing raw runs

Raw per-cell records go to ``--out`` (default ``runs/<scale>``, git-ignored) and are reused when present (resume);
tables, figures and ``summary.md`` go to ``--results`` (default ``results/<scale>``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from afkn.eval.cells import build_grid, merge_exps
from afkn.eval.config import load_eval_config
from afkn.eval.report import generate_report, load_records

ALL_EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5", "E6", "E8")  # E7 (CARLA) is not implemented


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", default="smoke", choices=["smoke", "small", "full"])
    ap.add_argument("--out", default=None, help="raw run records directory (default runs/<scale>)")
    ap.add_argument("--results", default=None, help="tables/figures directory (default results/<scale>)")
    ap.add_argument("--experiments", default=",".join(ALL_EXPERIMENTS), help="comma-separated subset of E1..E8")
    ap.add_argument("--report-only", action="store_true", help="skip running; rebuild the report from existing records")
    ap.add_argument("--overrides", default=None, help="JSON deep-merged into the scale block (e.g. '{\"seeds\": [0, 1]}')")
    args = ap.parse_args(argv)

    ec = load_eval_config(args.scale, overrides=json.loads(args.overrides) if args.overrides else None)
    out = Path(args.out or f"runs/{args.scale}")
    results = Path(args.results or f"results/{args.scale}")
    exps = tuple(e.strip() for e in args.experiments.split(",") if e.strip())
    unknown = set(exps) - set(ALL_EXPERIMENTS)
    if unknown:
        print(f"unknown or unimplemented experiments: {sorted(unknown)} (E7 needs CARLA and is not implemented)", file=sys.stderr)
        return 2

    if not args.report_only:
        from afkn.eval.runner import run_cells  # imports PyTorch

        cells = build_grid(ec.scale, exps)
        print(f"{len(cells)} cells, {len(merge_exps(cells))} unique runs")
        run_cells(cells, ec, out)
    records = load_records(out / "cells")
    if exps != ALL_EXPERIMENTS:
        records = [r for r in records if set(r.get("exps", [])) & set(exps)]
    files = generate_report(records, results, ec)
    print(f"wrote {len(files)} files under {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
