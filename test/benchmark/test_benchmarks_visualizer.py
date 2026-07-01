import os

import matplotlib
import pandas as pd

from benchmark import benchmarks_visualizer as vis

matplotlib.use("Agg")


def _make_plot_df(x_values):
    rows = []
    providers = ["liger", "torch"]
    for provider_idx, provider in enumerate(providers):
        for x in x_values:
            rows.append(
                {
                    "x_label": "X",
                    "x_value": x,
                    "metric_unit": "ms",
                    "kernel_provider": provider,
                    "y_value_20": 0.8 + provider_idx,
                    "y_value_50": 1.0 + provider_idx,
                    "y_value_80": 1.2 + provider_idx,
                    "gpu_name": "GPU-TEST",
                }
            )
    return pd.DataFrame(rows)


def test_build_data_source_suffix_default_and_non_default():
    assert vis.build_data_source_suffix(None) == ""
    assert vis.build_data_source_suffix(vis.DEFAULT_DATA_FILE) == ""
    assert vis.build_data_source_suffix("data/all_benchmark_data_cutile.csv") == "_all_benchmark_data_cutile"
    assert vis.build_data_source_suffix("/tmp/all_benchmark_data_cutedsl.csv") == "_all_benchmark_data_cutedsl"


def test_plot_data_bar_mode_numeric_x_with_source_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(vis, "VISUALIZATIONS_PATH", str(tmp_path))

    df = _make_plot_df([8, 16, 32])
    config = vis.VisualizationsConfig(
        kernel_name="rms_norm",
        metric_name="speed",
        kernel_operation_mode="forward",
        sweep_mode="token_length",
        data_file="data/all_benchmark_data_cutile.csv",
        plot_style="bar",
        overwrite=True,
    )

    vis.plot_data(df, config)

    expected = os.path.join(
        str(tmp_path),
        "rms_norm_speed_forward_token_length_all_benchmark_data_cutile.png",
    )
    assert os.path.exists(expected)


def test_plot_data_line_mode_categorical_x_default_source(tmp_path, monkeypatch):
    monkeypatch.setattr(vis, "VISUALIZATIONS_PATH", str(tmp_path))

    df = _make_plot_df(["cfg_small", "cfg_large"])
    config = vis.VisualizationsConfig(
        kernel_name="layer_norm",
        metric_name="memory",
        kernel_operation_mode="full",
        sweep_mode="model_config",
        data_file=None,
        plot_style="line",
        overwrite=True,
    )

    vis.plot_data(df, config)

    expected = os.path.join(
        str(tmp_path),
        "layer_norm_memory_full_model_config.png",
    )
    assert os.path.exists(expected)
