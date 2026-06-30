import os
import subprocess
import sys

import pytest

from benchmark.scripts import run_cutedsl_compare
from benchmark.scripts import run_cutile_compare


class _Completed:
    def __init__(self, returncode=0):
        self.returncode = returncode


def test_run_cutile_compare_invokes_two_runs_with_expected_env(monkeypatch):
    calls = []

    monkeypatch.setattr(sys, "argv", ["run_cutile_compare.py", "--kernel", "rms_norm", "--foo", "bar"]) 

    def fake_run(cmd, env, cwd):
        calls.append((cmd, env, cwd))
        return _Completed(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os.path, "isfile", lambda _: True)

    run_cutile_compare.main()

    assert len(calls) == 2

    for cmd, env, cwd in calls:
        assert cmd[:2] == [sys.executable, os.path.join(os.path.dirname(run_cutile_compare.__file__), "benchmark_rms_norm.py")]
        assert cmd[-2:] == ["--foo", "bar"]
        assert cwd == os.path.dirname(os.path.abspath(run_cutile_compare.__file__))
        assert env["LIGER_BENCH_TARGET"] == "cutile"

    assert calls[0][1]["LIGER_KERNEL_IMPL"] == ""
    assert calls[0][1]["LIGER_BENCH_PROVIDER_TAG"] == "liger_triton"
    assert calls[1][1]["LIGER_KERNEL_IMPL"] == "cutile"
    assert calls[1][1]["LIGER_BENCH_PROVIDER_TAG"] == "liger_cutile"


def test_run_cutedsl_compare_invokes_three_runs_with_expected_env(monkeypatch):
    calls = []

    monkeypatch.setattr(sys, "argv", ["run_cutedsl_compare.py", "--kernel", "rms_norm", "--abc", "xyz"])

    def fake_run(cmd, env, cwd):
        calls.append((cmd, env, cwd))
        return _Completed(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os.path, "isfile", lambda _: True)

    run_cutedsl_compare.main()

    assert len(calls) == 3

    expected_impls = ["", "cutile", "cutedsl"]
    expected_tags = ["liger_triton", "liger_cutile", "liger_cutedsl"]
    for idx, (_, env, _) in enumerate(calls):
        assert env["LIGER_BENCH_TARGET"] == "cutedsl"
        assert env["LIGER_KERNEL_IMPL"] == expected_impls[idx]
        assert env["LIGER_BENCH_PROVIDER_TAG"] == expected_tags[idx]


def test_compare_drivers_fail_when_benchmark_script_missing(monkeypatch):
    monkeypatch.setattr(os.path, "isfile", lambda _: False)

    monkeypatch.setattr(sys, "argv", ["run_cutile_compare.py", "--kernel", "rms_norm"])
    with pytest.raises(SystemExit, match="1"):
        run_cutile_compare.main()

    monkeypatch.setattr(sys, "argv", ["run_cutedsl_compare.py", "--kernel", "rms_norm"])
    with pytest.raises(SystemExit, match="1"):
        run_cutedsl_compare.main()


def test_compare_drivers_propagate_non_zero_returncode(monkeypatch):
    monkeypatch.setattr(os.path, "isfile", lambda _: True)

    monkeypatch.setattr(sys, "argv", ["run_cutile_compare.py", "--kernel", "rms_norm"])

    cutile_returns = iter([0, 7])

    def fake_cutile_run(cmd, env, cwd):
        del cmd, env, cwd
        return _Completed(next(cutile_returns))

    monkeypatch.setattr(subprocess, "run", fake_cutile_run)
    with pytest.raises(SystemExit, match="7"):
        run_cutile_compare.main()

    monkeypatch.setattr(sys, "argv", ["run_cutedsl_compare.py", "--kernel", "rms_norm"])

    cutedsl_returns = iter([0, 0, 9])

    def fake_cutedsl_run(cmd, env, cwd):
        del cmd, env, cwd
        return _Completed(next(cutedsl_returns))

    monkeypatch.setattr(subprocess, "run", fake_cutedsl_run)
    with pytest.raises(SystemExit, match="9"):
        run_cutedsl_compare.main()
