import pandas as pd

from paygap.plots import plot_study, to_percent
from paygap.study import run_study, summarize


def test_summarize_bias_rmse_coverage():
    runs = pd.DataFrame({
        "scenario": ["a"] * 2, "rep": [0, 1], "method": ["ols"] * 2,
        "est": [-0.04, -0.08], "lo": [-0.05, -0.09], "hi": [-0.03, -0.07],
        "true_direct": [-0.05] * 2, "true_total": [-0.10] * 2, "overlap": [None] * 2,
    })
    s = summarize(runs).iloc[0]
    assert abs(s["bias"] - (-0.01)) < 1e-12
    assert abs(s["rmse"] - ((0.01 ** 2 + 0.03 ** 2) / 2) ** 0.5) < 1e-12
    assert s["coverage"] == 0.5 and abs(s["bias_total"] - 0.04) < 1e-12


def test_small_study_and_plot(tmp_path):
    params = {"max_iter": 30, "max_leaf_nodes": 7}
    runs = run_study(["linear", "proxy"], reps=2, n=600, gap=-0.05, params=params, folds=2)
    assert len(runs) == 2 * 2 * 4
    assert set(summarize(runs)["method"]) == {"raw", "ols", "ml_ob", "dml"}
    plot_study(runs, tmp_path / "study.png")
    assert (tmp_path / "study.png").stat().st_size > 0


def test_to_percent():
    assert abs(to_percent(0.0)) < 1e-12 and abs(to_percent(-0.05) + 4.877) < 1e-3
