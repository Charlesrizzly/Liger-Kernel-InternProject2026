import importlib

import pytest
import torch

from test.transformers.test_rms_norm import BaseRMSNorm
from test.transformers.test_rms_norm import GemmaRMSNorm
from test.transformers.test_rms_norm import LlamaRMSNorm
from test.utils import assert_verbose_allclose
from test.utils import supports_bfloat16

BACKENDS = [
    "liger_kernel.ops.cutile.ops.rms_norm",
    "liger_kernel.ops.cutedsl.ops.rms_norm",
]


def _has_cuda_tile_runtime() -> bool:
    try:
        import cuda.tile  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="RMSNorm backend parity tests require CUDA"),
    pytest.mark.skipif(not _has_cuda_tile_runtime(), reason="RMSNorm backend parity tests require cuda-tile runtime"),
]


@pytest.mark.parametrize("backend_module", BACKENDS)
@pytest.mark.parametrize("shape", [(2, 8, 128), (2, 5, 127)])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 5e-4, 5e-5),
        pytest.param(
            torch.bfloat16,
            2e-1,
            2e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
@pytest.mark.parametrize(
    "reference, offset, casting_mode",
    [
        (LlamaRMSNorm, 0.0, "llama"),
        (GemmaRMSNorm, 1.0, "gemma"),
        (BaseRMSNorm, 0.0, "none"),
    ],
)
@pytest.mark.parametrize("in_place", [True, False])
@pytest.mark.parametrize("elementwise_affine", [True, False])
def test_backend_rms_norm_forward_backward_parity(
    backend_module,
    shape,
    dtype,
    atol,
    rtol,
    reference,
    offset,
    casting_mode,
    in_place,
    elementwise_affine,
):
    backend = importlib.import_module(backend_module)

    x = torch.randn(*shape, device="cuda", dtype=dtype)
    grad_out = torch.randn_like(x)

    x_ref = x.detach().clone().requires_grad_(True)
    x_backend = x.detach().clone().requires_grad_(True)

    ref_mod = reference(hidden_size=shape[-1], elementwise_affine=elementwise_affine).cuda().to(dtype)
    if elementwise_affine:
        w = ref_mod.weight.detach().clone().requires_grad_(True)
    else:
        w = None

    y_ref = ref_mod(x_ref)
    y_backend = backend.LigerRMSNormFunction.apply(x_backend, w, 1e-6, offset, casting_mode, in_place, None)

    y_ref.backward(grad_out, retain_graph=True)
    y_backend.backward(grad_out, retain_graph=True)

    assert_verbose_allclose(y_ref, y_backend, atol=atol, rtol=rtol)
    assert_verbose_allclose(x_ref.grad, x_backend.grad, atol=atol, rtol=rtol)
    if elementwise_affine:
        assert_verbose_allclose(ref_mod.weight.grad, w.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("backend_module", BACKENDS)
def test_backend_rms_norm_unsupported_hidden_size_raises(backend_module):
    backend = importlib.import_module(backend_module)

    # Only one row keeps memory usage tiny while still triggering the >65536 guard.
    hidden_size = 65537
    x = torch.randn(1, hidden_size, device="cuda", dtype=torch.float32)
    w = torch.ones(hidden_size, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="65536"):
        backend.rms_norm_forward(x, w, 1e-6, 0.0, "llama", None)
