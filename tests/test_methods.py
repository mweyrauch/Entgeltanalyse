import numpy as np
import pandas as pd

from paygap.generate import generate
from paygap.methods import dml_gap, linear_design, ml_ob_gap, ols_gap, raw_gap

PARAMS = {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 15}


def test_raw_gap_is_difference_of_means():
    df = pd.DataFrame({"female": [1, 1, 0, 0], "log_wage": [1.0, 2.0, 3.0, 5.0]})
    e = raw_gap(df)
    assert e.est == 1.5 - 4.0 and e.lo < e.est < e.hi


def test_ols_recovers_gap_in_linear_scenario():
    df, _ = generate(8000, "linear", gap=-0.05, seed=4)
    e = ols_gap(df)
    assert abs(e.est + 0.05) < 0.01 and e.lo < -0.05 < e.hi


def test_linear_design_has_intercept_and_no_gender():
    df, _ = generate(300, "linear", seed=0)
    X = linear_design(df)
    assert np.all(X[:, 0] == 1) and X.shape[0] == 300
    assert not np.allclose(X[:, 1:], df[["female"]].to_numpy())


def test_dml_recovers_gap_in_nonlinear_scenario():
    df, _ = generate(6000, "nonlinear", gap=-0.05, seed=5)
    e = dml_gap(df, PARAMS, folds=3, seed=0)
    assert abs(e.est + 0.05) < 0.015
    assert 0 <= e.extra["overlap"] <= 1


def test_no_gap_gives_estimates_near_zero():
    df, _ = generate(6000, "nonlinear", gap=0.0, seed=6)
    assert abs(dml_gap(df, PARAMS, folds=3).est) < 0.015
    assert abs(ml_ob_gap(df, PARAMS).est) < 0.02
