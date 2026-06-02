"""Run IPPO over several seeds (one config) and plot mean +- std bands.

Each seed is a separate subprocess (crash-isolated), writing its own
``results_dir/config_name/seed_<seed>.npz``. After all seeds finish, the
per-seed curves are aggregated into mean +- std plots in the same directory.

Example:
    python run_seeds_ippo.py --config-name spread --n-seeds 5 \
        --reward-mode spread --total-timesteps 2000000
Any extra IPPO flags can be forwarded after ``--``:
    python run_seeds_ippo.py --config-name spread -- --num-envs 8 --norm-returns true
"""

import argparse
import os
import subprocess
import sys

import ippo_cleanrl.metrics_scripts.metrics as metrics


def main():
    p = argparse.ArgumentParser(description="Multi-seed IPPO runner + aggregation")
    p.add_argument("--config-name", default="spread")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=1)
    p.add_argument("--reward-mode", default="spread")
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--results-dir", default="results_ippo")
    p.add_argument("--aggregate-only", action="store_true",
                   help="skip training; only re-aggregate existing seed files")
    p.add_argument("passthrough", nargs="*",
                   help="extra flags forwarded to the trainer (after --)")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    trainer = os.path.join(here, "ippo_simple_assignment.py")
    config_dir = os.path.join(args.results_dir, args.config_name)

    if not args.aggregate_only:
        for i in range(args.n_seeds):
            seed = args.base_seed + i
            cmd = [
                sys.executable, trainer,
                "--seed", str(seed),
                "--config-name", args.config_name,
                "--reward-mode", args.reward_mode,
                "--results-dir", args.results_dir,
                "--total-timesteps", str(args.total_timesteps),
            ] + args.passthrough
            print(f"\n=== seed {seed} ({i + 1}/{args.n_seeds}) ===")
            print("running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

    metrics.aggregate_and_plot(config_dir)


if __name__ == "__main__":
    main()
