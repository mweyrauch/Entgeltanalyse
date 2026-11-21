"""Schaetzer fuer die bereinigte Entgeltluecke (in Log-Punkten des Stundenentgelts).

Alle Schaetzer liefern ein `Estimate` mit Punktschaetzung und 95-%-Intervall.
- raw:   unbereinigte Luecke, Differenz der Mittelwerte.
- ols:   lineare Regression mit Geschlechtsindikator, heteroskedastie-robuste Fehler (HC1).
- ml_ob: ML-Oaxaca-Blinder. Ein Modell lernt die Entgeltstruktur der Maenner und sagt fuer
         jede Frau vorher, was ein Mann mit ihren Merkmalen verdienen wuerde. Der Mittelwert
         der Abweichungen ist der unerklaerte Teil. Das Intervall beruecksichtigt nur das
         Rauschen der Frauen, nicht die Unsicherheit des Modells.
- dml:   Double Machine Learning im partiell linearen Modell y = theta * d + g(X) + e mit
         Cross-Fitting (Chernozhukov et al. 2018).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

FEATURES = ["job", "location", "age", "experience", "parttime", "level"]
CATEGORICAL = ["job", "location"]
Z95 = 1.959963984540054


@dataclass
class Estimate:
    method: str
    est: float
    lo: float
    hi: float
    extra: dict | None = None


def _ci(method, est, se, extra=None) -> Estimate:
    return Estimate(method, float(est), float(est - Z95 * se), float(est + Z95 * se), extra)


def raw_gap(df: pd.DataFrame) -> Estimate:
    y, d = df["log_wage"].to_numpy(), df["female"].to_numpy() == 1
    est = y[d].mean() - y[~d].mean()
    se = np.sqrt(y[d].var(ddof=1) / d.sum() + y[~d].var(ddof=1) / (~d).sum())
    return _ci("raw", est, se)


def linear_design(df: pd.DataFrame) -> np.ndarray:
    """Designmatrix fuer die lineare Regression: Dummies fuer Jobfamilie, Standort und
    Entgeltstufe, dazu Alter, Erfahrung und Teilzeit (ohne Geschlecht)."""
    dummies = pd.get_dummies(df[["job", "location"]].astype(str), drop_first=True)
    levels = pd.get_dummies(df["level"].astype(str), prefix="level", drop_first=True)
    num = df[["age", "experience", "parttime"]]
    X = pd.concat([num, dummies, levels], axis=1).astype(float).to_numpy()
    return np.column_stack([np.ones(len(df)), X])


def ols_gap(df: pd.DataFrame) -> Estimate:
    X = np.column_stack([df["female"].to_numpy(float), linear_design(df)])
    y = df["log_wage"].to_numpy()
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    meat = (X * resid[:, None] ** 2).T @ X
    cov = XtX_inv @ meat @ XtX_inv * n / (n - k)                   # HC1
    return _ci("ols", beta[0], np.sqrt(cov[0, 0]))


def ml_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def _regressor(params: dict, seed: int):
    return HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=seed,
                                         **params)


def _classifier(params: dict, seed: int):
    return HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=seed,
                                          **params)


def ml_ob_gap(df: pd.DataFrame, params: dict | None = None, seed: int = 0) -> Estimate:
    params = params or {}
    X, y = ml_features(df), df["log_wage"].to_numpy()
    d = df["female"].to_numpy() == 1
    model = _regressor(params, seed).fit(X[~d], y[~d])
    diff = y[d] - model.predict(X[d])
    return _ci("ml_ob", diff.mean(), diff.std(ddof=1) / np.sqrt(d.sum()))


def dml_gap(df: pd.DataFrame, params: dict | None = None, folds: int = 5, seed: int = 0,
            clip: float = 0.01) -> Estimate:
    """Partiell lineares Modell mit Cross-Fitting.

    g(X) = E[y | X] und m(X) = P(Frau | X) werden je Fold auf den uebrigen Folds gelernt.
    theta ergibt sich aus der Regression der Restgroessen y - g(X) auf d - m(X).
    extra["overlap"]: Anteil der Personen mit m(X) ausserhalb [0.05, 0.95]; dort gibt es kaum
    vergleichbare Personen des anderen Geschlechts.
    """
    params = params or {}
    X, y = ml_features(df), df["log_wage"].to_numpy()
    d = df["female"].to_numpy().astype(float)
    g_hat, m_hat = np.empty(len(df)), np.empty(len(df))
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, d):
        g_hat[test] = _regressor(params, seed).fit(X.iloc[train], y[train]).predict(X.iloc[test])
        clf = _classifier(params, seed).fit(X.iloc[train], d[train])
        m_hat[test] = clf.predict_proba(X.iloc[test])[:, 1]
    m_hat = np.clip(m_hat, clip, 1 - clip)
    u, v = y - g_hat, d - m_hat
    theta = (v @ u) / (v @ v)
    psi = (u - theta * v) * v
    se = np.sqrt(np.mean(psi ** 2) / np.mean(v ** 2) ** 2 / len(df))
    overlap = float(np.mean((m_hat < 0.05) | (m_hat > 0.95)))
    return _ci("dml", theta, se, {"overlap": overlap})


def all_estimates(df: pd.DataFrame, params: dict | None = None, folds: int = 5,
                  seed: int = 0) -> list[Estimate]:
    return [raw_gap(df), ols_gap(df), ml_ob_gap(df, params, seed),
            dml_gap(df, params, folds, seed)]
