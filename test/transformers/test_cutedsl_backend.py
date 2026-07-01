"""Verify LIGER_KERNEL_IMPL=cutedsl routes ops and transformers to cuTeDSL implementations."""

import importlib
import os

import pytest
import torch

CUTEDSL_PREFIX = "liger_kernel.ops.cutedsl."

# Transformer modules bind liger_kernel.ops.* at import time (geglu != GELUMul name).
TRANSFORMER_MODULES = {
    "LigerRMSNormFunction": "liger_kernel.transformers.rms_norm",
}


def _has_cuda_tile_runtime() -> bool:
    try:
        import cuda.tile  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="cuTeDSL backend requires CUDA",
    ),
    pytest.mark.skipif(
        not _has_cuda_tile_runtime(),
        reason="cuTeDSL backend requires cuda-tile runtime",
    ),
    pytest.mark.skipif(
        os.environ.get("LIGER_KERNEL_IMPL", "").strip().lower() != "cutedsl",
        reason="cuTeDSL backend selection test requires LIGER_KERNEL_IMPL=cutedsl",
    ),
]


def test_liger_kernel_impl_cutedsl_routes():
    import liger_kernel.ops as ops
    import liger_kernel.ops.cutedsl.ops as cutedsl_ops

    failures = []

    for name in cutedsl_ops.__all__:
        if not name.endswith("Function"):
            continue
        cls = getattr(ops, name)
        if not cls.__module__.startswith(CUTEDSL_PREFIX):
            failures.append(f"liger_kernel.ops.{name} -> {cls.__module__}")

    for name, module_path in TRANSFORMER_MODULES.items():
        cls = getattr(importlib.import_module(module_path), name)
        if not cls.__module__.startswith(CUTEDSL_PREFIX):
            failures.append(f"{module_path}.{name} -> {cls.__module__}")

    assert not failures, "expected cuTeDSL routing:\n" + "\n".join(failures)