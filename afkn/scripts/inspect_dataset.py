"""Build the federated dataset in memory and print its summary (writes nothing to disk).

    python -m afkn.scripts.inspect_dataset [--config path.yaml] [--seed N] [--json]
"""

from __future__ import annotations

import argparse
import json
import time

from afkn.data.partition import build_federated_dataset
from afkn.sim.config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="YAML config (default: packaged default.yaml)")
    ap.add_argument("--seed", type=int, default=None, help="override cfg.seed")
    ap.add_argument("--json", action="store_true", help="print the raw summary as JSON")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    t0 = time.perf_counter()
    ds = build_federated_dataset(cfg, args.seed)
    elapsed = time.perf_counter() - t0
    s = ds.summary()
    if args.json:
        print(json.dumps(s, indent=2))
        return 0

    print(f"seed={s['seed']} steps={s['steps']} dt={s['dt']} model={s['motion_model']}  built in {elapsed:.1f}s")
    print(f"config_hash={s['config_hash'][:16]}  fingerprint={s['fingerprint'][:16]}")
    print(f"global_test n={s['global_test_n']} {s['global_test_profile_counts']}")
    print(f"trusted     n={s['trusted_n']} {s['trusted_profile_counts']}")
    print(f"{'id':>2} {'profile':<13} {'fault':<14} {'train/val/test':<14} {'sigma':>6} {'gnss_avail':>10} {'urban/hwy/tunnel':<18} {'v_mean':>6}")
    for r in s["clients"]:
        fr = "/".join(f"{v:.2f}" for v in r["seg_frac_urban_highway_tunnel"])
        tvt = f"{r['n_train']}/{r['n_val']}/{r['n_test']}"
        print(
            f"{r['client']:>2} {r['profile']:<13} {str(r['fault_mode']):<14} {tvt:<14} "
            f"{r['gnss_sigma']:>6.2f} {r['gnss_available']:>10.3f} {fr:<18} {r['mean_speed']:>6.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
