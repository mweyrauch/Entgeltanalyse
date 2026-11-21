"""Simulationsstudie: wie gut trifft jede Methode die eingebaute Luecke?"""
from __future__ import annotations

import pandas as pd

from .generate import generate
from .methods import all_estimates


def run_study(scenarios, reps: int, n: int, gap: float, params: dict, folds: int,
              seed: int = 0, progress=None) -> pd.DataFrame:
    """Eine Zeile je Szenario, Wiederholung und Methode."""
    rows = []
    for s in scenarios:
        for r in range(reps):
            df, truth = generate(n, s, gap, seed=seed + r)
            for e in all_estimates(df, params, folds, seed=seed + r):
                rows.append({"scenario": s, "rep": r, "method": e.method, "est": e.est,
                             "lo": e.lo, "hi": e.hi, "true_direct": truth["direct"],
                             "true_total": truth["total"],
                             "overlap": (e.extra or {}).get("overlap")})
            if progress:
                progress(s, r)
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Verzerrung, RMSE und Intervallabdeckung gegenueber der direkten Luecke; dazu die
    Verzerrung gegenueber der Gesamtluecke (unterscheidet sich nur im Szenario proxy)."""
    r = runs.assign(err=runs["est"] - runs["true_direct"],
                    err_total=runs["est"] - runs["true_total"],
                    covered=(runs["lo"] <= runs["true_direct"]) & (runs["true_direct"] <= runs["hi"]),
                    width=runs["hi"] - runs["lo"])
    g = r.groupby(["scenario", "method"], sort=False)
    out = pd.DataFrame({
        "mean_est": g["est"].mean(),
        "bias": g["err"].mean(),
        "rmse": g["err"].apply(lambda e: (e ** 2).mean() ** 0.5),
        "coverage": g["covered"].mean(),
        "ci_width": g["width"].mean(),
        "bias_total": g["err_total"].mean(),
        "reps": g.size(),
    })
    return out.reset_index()
