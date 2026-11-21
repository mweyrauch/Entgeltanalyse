"""Analysiert einen einzelnen synthetischen Datensatz und zeigt alle Schaetzer.

Beispiel:
    python scripts/analyze.py --scenario nonlinear --seed 1
    python scripts/analyze.py --csv data/beschaeftigte.csv   # eigene Daten im selben Format
"""
from __future__ import annotations

import argparse

import pandas as pd

from paygap.config import load_config
from paygap.generate import generate
from paygap.methods import FEATURES, all_estimates
from paygap.plots import METHOD_LABELS, to_percent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scenario", default="nonlinear")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int)
    ap.add_argument("--csv", help="eigene Daten statt Simulation")
    ap.add_argument("--save", help="simulierten Datensatz als CSV speichern")
    a = ap.parse_args()
    cfg = load_config(a.config, n=a.n)
    st, me = cfg["study"], cfg["methods"]

    if a.csv:
        df, truth = pd.read_csv(a.csv), None
        for c in ("job", "location"):
            df[c] = df[c].astype("category")
    else:
        df, truth = generate(st["n"], a.scenario, st["gap"], seed=a.seed)
        if a.save:
            df.to_csv(a.save, index=False)
            print(f"Datensatz gespeichert in {a.save}")
    missing = set(FEATURES + ["female", "log_wage"]) - set(df.columns)
    if missing:
        raise SystemExit(f"Fehlende Spalten: {sorted(missing)}")

    print(f"{len(df)} Beschäftigte, Frauenanteil {df['female'].mean():.0%}")
    if truth:
        print(f"Szenario {a.scenario}: wahre direkte Lücke {to_percent(truth['direct']):.1f} %, "
              f"Gesamtlücke {to_percent(truth['total']):.1f} %")
    print(f"\n{'Methode':<26}{'Lücke':>9}   95-%-Intervall")
    for e in all_estimates(df, me["model_params"], me["folds"], seed=a.seed):
        lo, est, hi = to_percent([e.lo, e.est, e.hi])
        print(f"{METHOD_LABELS[e.method]:<26}{est:>8.1f} %   [{lo:.1f} %, {hi:.1f} %]")
        if e.extra and "overlap" in e.extra:
            overlap = e.extra["overlap"]
    print(f"\nKaum vergleichbar (P(Frau | Merkmale) < 5 % oder > 95 %): {overlap:.0%} "
          "der Beschäftigten")


if __name__ == "__main__":
    main()
