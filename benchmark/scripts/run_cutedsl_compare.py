"""Run a kernel benchmark three times (Triton + CuTile + CuteDSL) into one CSV.

Workflow:
    python scripts/run_cutedsl_compare.py --kernel rms_norm [benchmark args...]

This driver spawns the per-kernel benchmark script in three subprocesses with
different env vars, so all four series (liger_triton / liger_cutile /
liger_cutedsl / huggingface or torch) land in
`benchmark/data/all_benchmark_data_cutedsl.csv` under distinct
`kernel_provider` values, ready for direct plotting via:

    python ../benchmarks_visualizer.py \
        --kernel-name <name> --metric-name speed \
        --data-file data/all_benchmark_data_cutedsl.csv
"""

import argparse
import os
import subprocess
import sys

CUTEDSL_ENABLED_KERNELS = [
    "rms_norm",
]


def main():
    parser = argparse.ArgumentParser(
        description="Compare Triton vs CuTile vs CuteDSL Liger kernels in one CSV.",
        # Unknown args are forwarded to the underlying benchmark script.
    )
    parser.add_argument(
        "--kernel",
        required=True,
        choices=CUTEDSL_ENABLED_KERNELS,
        help="Kernel to compare. Must have a CuteDSL backend.",
    )
    args, passthrough = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bench_script = os.path.join(script_dir, f"benchmark_{args.kernel}.py")
    if not os.path.isfile(bench_script):
        print(f"error: benchmark script not found: {bench_script}", file=sys.stderr)
        sys.exit(1)

    # All runs target the same _cutedsl.csv; provider_tag disambiguates the
    # "liger" rows so they don't overwrite each other on the dedup key.
    runs = [
        ("triton baseline", {"LIGER_KERNEL_IMPL": "", "LIGER_BENCH_PROVIDER_TAG": "liger_triton"}),
        ("cutile", {"LIGER_KERNEL_IMPL": "cutile", "LIGER_BENCH_PROVIDER_TAG": "liger_cutile"}),
        ("cutedsl", {"LIGER_KERNEL_IMPL": "cutedsl", "LIGER_BENCH_PROVIDER_TAG": "liger_cutedsl"}),
    ]

    for label, run_env in runs:
        print(f"\n========== {args.kernel}: {label} ==========\n", flush=True)
        env = {**os.environ, "LIGER_BENCH_TARGET": "cutedsl", **run_env}
        result = subprocess.run(
            [sys.executable, bench_script, *passthrough],
            env=env,
            cwd=script_dir,
        )
        if result.returncode != 0:
            print(f"error: {label} run failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()