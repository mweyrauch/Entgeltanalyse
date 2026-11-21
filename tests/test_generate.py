import numpy as np
import pytest

from paygap.generate import SCENARIOS, generate


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_columns_and_reproducibility(scenario):
    a, ta = generate(500, scenario, seed=3)
    b, tb = generate(500, scenario, seed=3)
    assert list(a.columns) == ["female", "job", "location", "age", "experience", "parttime",
                               "level", "log_wage", "hourly_wage"]
    assert a.equals(b) and ta == tb
    assert a["level"].between(1, 5).all() and (a["experience"] >= 0).all()


def test_direct_gap_equals_parameter_and_total_only_differs_for_proxy():
    for s in ("linear", "nonlinear"):
        _, t = generate(2000, s, gap=-0.07, seed=1)
        assert t["direct"] == -0.07 and abs(t["total"] + 0.07) < 1e-12
    _, t = generate(2000, "proxy", gap=-0.07, seed=1)
    assert t["total"] < -0.07 - 0.02          # seltenere Befoerderung vergroessert die Luecke


def test_proxy_only_lowers_levels_of_women():
    base, _ = generate(3000, "nonlinear", seed=2)
    proxy, _ = generate(3000, "proxy", seed=2)
    men = base["female"] == 0
    assert (base.loc[men, "level"] == proxy.loc[men, "level"]).all()
    assert proxy.loc[~men, "level"].mean() < base.loc[~men, "level"].mean()


def test_zero_gap_means_identical_wage_function():
    df, t = generate(1000, "linear", gap=0.0, seed=0)
    assert t == {"direct": 0.0, "total": 0.0}
    assert np.isfinite(df["log_wage"]).all()
