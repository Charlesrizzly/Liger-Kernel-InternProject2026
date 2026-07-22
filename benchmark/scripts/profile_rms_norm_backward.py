"""Profile one warmed-up RMSNorm backward pass with Nsight Systems or Compute.

Select the implementation before process startup:

    LIGER_KERNEL_IMPL=cutedsl python profile_rms_norm_backward.py
    LIGER_KERNEL_IMPL= python profile_rms_norm_backward.py
"""

import argparse
import os

import torch

from liger_kernel.ops import rms_norm_backward
from liger_kernel.ops import rms_norm_forward


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--weight-dtype",
        choices=["bfloat16", "float32"],
        default="float32",
        help="RMSNorm weight dtype; benchmarks use float32 weights by default",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RMSNorm profiling")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")

    dtype = getattr(torch, args.dtype)
    weight_dtype = getattr(torch, args.weight_dtype)
    device = torch.device("cuda")
    implementation = os.environ.get("LIGER_KERNEL_IMPL", "").strip() or "triton"

    torch.manual_seed(42)
    x = torch.randn(args.rows, args.hidden_size, device=device, dtype=dtype)
    w = torch.randn(args.hidden_size, device=device, dtype=weight_dtype)
    dy_template = torch.randn_like(x)

    def forward():
        return rms_norm_forward(x, w, 1e-6, 0.0, "llama", None)

    def backward(dy, forward_state):
        _, x_2d, rstd, block_size, num_warps, casting_mode = forward_state
        return rms_norm_backward(
            dy,
            x_2d,
            w,
            rstd,
            0.0,
            casting_mode,
            block_size,
            num_warps,
            True,
            None,
        )

    for _ in range(args.warmup):
        backward(dy_template.clone(), forward())
    torch.cuda.synchronize()

    forward_state = forward()
    profile_dys = [dy_template.clone() for _ in range(args.iterations)]
    torch.cuda.synchronize()

    label = (
        f"rms_norm_backward_{implementation}_m{args.rows}_n{args.hidden_size}"
        f"_w{args.weight_dtype}_iters{args.iterations}"
    )
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    torch.cuda.nvtx.range_push(label)
    for dy in profile_dys:
        dx, dw = backward(dy, forward_state)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    cudart.cudaProfilerStop()

    print(
        f"profiled {label}: dx={tuple(dx.shape)}, dw={tuple(dw.shape)}, "
        f"sample=({dx[0, 0].item():.6f}, {dw[0].item():.6f})"
    )


if __name__ == "__main__":
    main()
