#!/usr/bin/env python3
"""Microbenchmark fused vs split RMSNorm backward for one shape.

Usage:
  python microbench_fused_vs_split.py --seq 4096 --dtype bfloat16 --reps 20

This script runs three configurations:
 - default (fast path allowed)
 - force split fallback (LIGER_RMS_FORCE_SPLIT_BWD=1)
 - disable fast vector path (LIGER_RMS_FORCE_NO_FAST=1)

It prints median timings for backward and full runs and emits simple JSON logs.
"""
import argparse
import json
import os
import time

try:
    import torch
    import triton
except Exception:
    print("torch/triton required to run microbench. Exiting.")
    raise

from benchmark.scripts.benchmark_rms_norm import setup_rms_norm
from benchmark.scripts.utils import run_speed_benchmark


def run_one(seq_len, hidden, dtype, reps, mode, env_overrides=None):
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    # Apply env overrides in-process for our module
    for k, v in (env_overrides or {}).items():
        os.environ[k] = v

    # Build input
    class Dummy:
        extra_benchmark_config = {"hidden_size": hidden, "dtype": getattr(torch, dtype), "eps": 1e-6, "bsz": 1, "seq_len": seq_len}
    inp = type('I', (), {})()
    inp.x = seq_len
    inp.kernel_provider = 'liger'
    inp.kernel_operation_mode = mode
    inp.extra_benchmark_config = Dummy.extra_benchmark_config

    setup_out = setup_rms_norm(inp)
    fwd_fn = lambda: setup_out[1](setup_out[0]) if hasattr(setup_out[1], '__call__') else None
    # But setup_rms_norm returns x, layer; run_speed_benchmark expects setup_fn
    # We'll reuse run_speed_benchmark by wrapping
    def fwd():
        x, layer = setup_out
        return layer(x)

    result = run_speed_benchmark(fwd, mode if mode else 'full', [setup_out[0]], rep=reps)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', type=int, default=4096)
    p.add_argument('--hidden', type=int, default=4096)
    p.add_argument('--dtype', type=str, default='bfloat16')
    p.add_argument('--reps', type=int, default=20)
    args = p.parse_args()

    configs = [
        ("default", {}),
        ("force_split", {"LIGER_RMS_FORCE_SPLIT_BWD": "1"}),
        ("no_fast", {"LIGER_RMS_FORCE_NO_FAST": "1"}),
    ]

    out = {}
    for name, env in configs:
        print(f"Running config: {name} env={env}")
        # backward
        os.environ.update(env)
        try:
            x, layer = setup_rms_norm(type('C', (), {'x': args.seq, 'kernel_provider': 'liger', 'kernel_operation_mode': 'backward', 'extra_benchmark_config': {'hidden_size': args.hidden, 'dtype': getattr(torch, args.dtype), 'eps': 1e-6, 'bsz': 1, 'seq_len': args.seq}}))
            def fwd():
                return layer(x)
            ms_50, ms_20, ms_80 = triton.testing.do_bench(
                lambda: (layer(x)), grad_to_none=[x], rep=args.reps, quantiles=[0.5, 0.2, 0.8]
            )
            out[name] = {'backward_median_ms': ms_50}
            print(f"{name} backward median: {ms_50:.6f} ms")
        except Exception as e:
            print(f"Error running {name}: {e}")
            out[name] = {'error': str(e)}

    print('\nSummary JSON:')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
