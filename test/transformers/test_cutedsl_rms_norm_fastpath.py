"""CPU-only checks for CuTe DSL RMSNorm fast-path launch policy."""

import importlib.util

from pathlib import Path

import pytest

_HELPERS_PATH = Path(__file__).parents[2] / "src/liger_kernel/ops/cutedsl/ops/rms_norm_fastpath.py"
_spec = importlib.util.spec_from_file_location("rms_norm_fastpath_test_helpers", _HELPERS_PATH)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
fast_path_vector_width = _helpers.fast_path_vector_width
persistent_strip_count = _helpers.persistent_strip_count
supports_global_cp_async = _helpers.supports_global_cp_async


def test_fast_path_vector_width_uses_largest_participating_element():
    assert fast_path_vector_width(2, 2) == 8
    assert fast_path_vector_width(2, 4, 2) == 4
    assert fast_path_vector_width(4, 4) == 4


def test_fast_path_vector_width_rejects_non_vectorizable_size():
    with pytest.raises(ValueError):
        fast_path_vector_width(3)


def test_global_cp_async_requires_full_16_byte_copy_for_every_staged_tensor():
    assert supports_global_cp_async(8, 2, 2)
    assert supports_global_cp_async(4, 4, 4)
    assert not supports_global_cp_async(4, 2, 2)


@pytest.mark.parametrize(
    ("capability", "n_cols", "expected"),
    [
        ((10, 0), 2048, 4 * 132),
        ((10, 0), 4096, 2 * 132),
        ((9, 0), 2048, 4 * 132),
        ((9, 0), 4096, 2 * 132),
        ((9, 0), 8192, 132),
        ((8, 9), 4096, 132),
    ],
)
def test_persistent_strip_count_arch_heuristic(capability, n_cols, expected):
    assert persistent_strip_count(132, n_cols, 10_000, capability) == expected


def test_persistent_strip_count_is_capped_at_rows():
    assert persistent_strip_count(132, 2048, 37, (10, 0)) == 37
