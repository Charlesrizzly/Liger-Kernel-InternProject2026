from types import SimpleNamespace

import pytest
import torch

import liger_kernel.ops.modulated_rms_norm as mod_rms_norm_ops
import liger_kernel.ops.rms_norm as rms_norm_ops
import liger_kernel.ops.swiglu as swiglu_ops
import liger_kernel.ops.utils as ops_utils


class _FakeDTensor:
    def __init__(self, tensor, device_mesh="mesh", placements=("shard",)):
        self._tensor = tensor
        self.device_mesh = device_mesh
        self.placements = placements

    def full_tensor(self):
        return self._tensor

    def to_local(self):
        return self._tensor

    @staticmethod
    def from_local(local_tensor, device_mesh, placements):
        return _FakeDTensor(local_tensor, device_mesh=device_mesh, placements=placements)


class _FakeDistributedTensorModule:
    DTensor = _FakeDTensor

    @staticmethod
    def distribute_tensor(tensor, device_mesh, placements):
        return _FakeDTensor(tensor, device_mesh=device_mesh, placements=placements)


def _make_ctx():
    ctx = SimpleNamespace()

    def save_for_backward(*tensors):
        ctx.saved_tensors = tensors

    ctx.save_for_backward = save_for_backward
    return ctx


def test_is_dtensor_helper_uses_shared_detection_path(monkeypatch):
    monkeypatch.setattr(ops_utils, "torch_distributed_tensor", _FakeDistributedTensorModule)
    assert ops_utils.is_dtensor(_FakeDTensor(torch.ones(2)))
    assert not ops_utils.is_dtensor(torch.ones(2))

    monkeypatch.setattr(ops_utils, "torch_distributed_tensor", None)
    assert not ops_utils.is_dtensor(_FakeDTensor(torch.ones(2)))


@pytest.mark.parametrize("has_weight", [True, False])
def test_rms_norm_dtensor_input_and_grad_paths(monkeypatch, has_weight):
    seen = {}

    def fake_forward(x, w, eps, offset, casting_mode, row_mode):
        del eps, offset, casting_mode, row_mode
        seen["forward_x_is_tensor"] = isinstance(x, torch.Tensor)
        seen["forward_w_is_expected"] = (w is None) if (not has_weight) else isinstance(w, torch.Tensor)
        n_rows = x.view(-1, x.shape[-1]).shape[0]
        return x + 1.0, x.view(-1, x.shape[-1]), torch.ones(n_rows), 128, None, 0

    def fake_backward(dy, x, w, rstd, offset, casting_mode, block_size, num_warps, in_place, row_mode):
        del x, w, rstd, offset, casting_mode, block_size, num_warps, in_place, row_mode
        seen["backward_dy_is_tensor"] = isinstance(dy, torch.Tensor)
        dw = torch.zeros(4) if has_weight else None
        return dy, dw

    monkeypatch.setattr(rms_norm_ops, "is_dtensor", lambda value: isinstance(value, _FakeDTensor))
    monkeypatch.setattr(rms_norm_ops, "rms_norm_forward", fake_forward)
    monkeypatch.setattr(rms_norm_ops, "rms_norm_backward", fake_backward)

    x = torch.randn(2, 3, 4)
    x_dt = _FakeDTensor(x)
    w = torch.randn(4) if has_weight else None

    ctx = _make_ctx()
    y = rms_norm_ops.LigerRMSNormFunction.forward(ctx, x_dt, w, 1e-6, 0.0, "llama", False, None)
    assert isinstance(y, torch.Tensor)
    assert seen["forward_x_is_tensor"]
    assert seen["forward_w_is_expected"]

    _ = rms_norm_ops.LigerRMSNormFunction.backward(ctx, _FakeDTensor(torch.ones_like(y)))
    assert seen["backward_dy_is_tensor"]


@pytest.mark.parametrize("has_weight", [True, False])
@pytest.mark.parametrize("has_shift", [True, False])
def test_modulated_rms_norm_dtensor_input_and_grad_paths(monkeypatch, has_weight, has_shift):
    seen = {}

    def fake_forward(x, w, scale, shift, eps, offset, casting_mode):
        del eps, offset, casting_mode
        seen["forward_x_is_tensor"] = isinstance(x, torch.Tensor)
        seen["forward_scale_is_tensor"] = isinstance(scale, torch.Tensor)
        seen["forward_shift_matches"] = (shift is None) if (not has_shift) else isinstance(shift, torch.Tensor)
        seen["forward_w_matches"] = (w is None) if (not has_weight) else isinstance(w, torch.Tensor)
        n_rows = x.view(-1, x.shape[-1]).shape[0]
        return x + scale.mean(), torch.ones(n_rows), 128, None, 0, 1

    def fake_backward(dy, x, w, scale, shift, rstd, offset, casting_mode, block_size, num_warps, rows_per_modulation, in_place):
        del x, w, scale, shift, rstd, offset, casting_mode, block_size, num_warps, rows_per_modulation, in_place
        seen["backward_dy_is_tensor"] = isinstance(dy, torch.Tensor)
        dw = torch.zeros(4) if has_weight else None
        dscale = torch.zeros(4)
        dshift = torch.zeros(4) if has_shift else None
        return dy, dw, dscale, dshift

    monkeypatch.setattr(mod_rms_norm_ops, "is_dtensor", lambda value: isinstance(value, _FakeDTensor))
    monkeypatch.setattr(mod_rms_norm_ops, "modulated_rms_norm_forward", fake_forward)
    monkeypatch.setattr(mod_rms_norm_ops, "modulated_rms_norm_backward", fake_backward)

    x = torch.randn(2, 3, 4)
    x_dt = _FakeDTensor(x)
    w = torch.randn(4) if has_weight else None
    scale = torch.randn(4)
    shift = torch.randn(4) if has_shift else None

    ctx = _make_ctx()
    y = mod_rms_norm_ops.LigerModulatedRMSNormFunction.forward(ctx, x_dt, w, scale, shift, 1e-6, 0.0, "llama", False)
    assert isinstance(y, torch.Tensor)
    assert seen["forward_x_is_tensor"]
    assert seen["forward_scale_is_tensor"]
    assert seen["forward_shift_matches"]
    assert seen["forward_w_matches"]

    _ = mod_rms_norm_ops.LigerModulatedRMSNormFunction.backward(ctx, _FakeDTensor(torch.ones_like(y)))
    assert seen["backward_dy_is_tensor"]


def test_swiglu_dtensor_flow_preserves_distributed_wrapping(monkeypatch):
    monkeypatch.setattr(swiglu_ops, "torch_distributed_tensor", _FakeDistributedTensorModule)
    monkeypatch.setattr(swiglu_ops, "is_dtensor", lambda value: isinstance(value, _FakeDTensor))
    monkeypatch.setattr(swiglu_ops, "swiglu_forward", lambda a, b, g: (a, b, a + b + g))
    monkeypatch.setattr(swiglu_ops, "swiglu_backward", lambda a, b, dc, g: (dc + a + g, dc + b + g))

    ctx = _make_ctx()
    a = torch.ones(2, 3)
    b_dt = _FakeDTensor(torch.full((2, 3), 2.0), device_mesh="mesh0", placements=("replicate",))

    out = swiglu_ops.LigerSiLUMulFunction.forward(ctx, a, b_dt, 1.5, 2.0)
    assert isinstance(out, _FakeDTensor)
    assert out.device_mesh == "mesh0"
    assert out.placements == ("replicate",)

    grad_dt = _FakeDTensor(torch.ones_like(out.to_local()), device_mesh="mesh0", placements=("replicate",))
    da, db, _, _ = swiglu_ops.LigerSiLUMulFunction.backward(ctx, grad_dt)
    assert isinstance(da, _FakeDTensor)
    assert isinstance(db, _FakeDTensor)
    assert da.device_mesh == "mesh0"
    assert db.device_mesh == "mesh0"
