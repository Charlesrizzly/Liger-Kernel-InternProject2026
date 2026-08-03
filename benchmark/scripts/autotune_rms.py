#!/usr/bin/env python3
"""Simple autotuner for RMSNorm CuTe DSL fused backward geometry.

This is a best-effort, lightweight tuner that runs small microbenchmarks (if
torch is available) for a set of (num_warps, bucket_size, force_split) options
and records the best-performing tuple per device/shape. It intentionally avoids
heavy framework dependencies and writes a JSON file with suggested environment
variables to use for production runs.

Usage:
    python autotune_rms.py --shapes "1024,2048,4096" --trials 3 --out tune_rms.json

If torch isn't importable the script exits with an explanatory message.
"""
import argparse
import json
import os
import time

try:
    import torch
except Exception:
    print("torch not found; autotuner requires torch + CUDA to run.")
    raise

from subprocess import run, CalledProcessError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_SCRIPT = os.path.join(SCRIPT_DIR, "run_cutedsl_compare.py")


def bench_with_env(env, source, logpath):
    env_full = {**os.environ, **env}
    cmd = ["python", RUN_SCRIPT, "--kernel", "rms_norm", "--source", source]
    with open(logpath, "w") as out:
        p = run(cmd, env=env_full, cwd=SCRIPT_DIR)
        return p.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="1024,2048", help="Comma list of shapes to test (hidden dims)")
    p.add_argument("--trials", type=int, default=2, help="Number of repetitions per config (warm+measured)")
    p.add_argument("--out", default="rms_autotune.json", help="Output JSON with best configs")
    p.add_argument("--source", default="auto", help="Label for CSV outputs (gpu name)")
    args = p.parse_args()

    shapes = [int(s) for s in args.shapes.split(",") if s.strip()]

    # Parameter grid (small to keep runs short). These are suggested starting
    # points; expand only after verifying runtime cost per trial.
    warp_options = [4, 6, 8]
    bucket_options = [0, 16, 32]
    force_split_options = [0, 1]

    results = {}
    for shape in shapes:
        best = {"time": 1e18, "config": None}
        for w in warp_options:
            for b in bucket_options:
                for fs in force_split_options:
                    env = {
                        "LIGER_RMS_DEBUG": "1",
                        "LIGER_RMS_COMPILE_BUCKET": str(b),
                        "LIGER_RMS_FORCE_SPLIT_BWD": str(fs),
                        # allow an explicit warp hint
                        "LIGER_RMS_BACKWARD_WARPS": str(w),
                    }
                    log = f"/tmp/rms_tune_{shape}_{w}_{b}_{fs}.log"
                    print(f"Testing shape={shape} w={w} bucket={b} force_split={fs} -> log={log}")
                    start = time.time()
                    ok = bench_with_env(env, args.source, log)
                    dur = time.time() - start
                    if not ok:
                        print(f"Run failed for config {env}; skipping")
                        continue
                    # crude: use wall time as proxy (CSV parsing is heavier to implement here)
                    print(f"Config completed in {dur:.2f}s")
                    if dur < best["time"]:
                        best = {"time": dur, "config": {"warps": w, "bucket": b, "force_split": fs}}
        results[shape] = best

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote tuning results to {args.out}")


if __name__ == '__main__':
    main()
