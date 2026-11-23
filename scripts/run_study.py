"""Simulationsstudie ueber alle Szenarien.

Beispiel:
    python scripts/run_study.py                    # 100 Wiederholungen, ca. 10-15 Minuten
    python scripts/run_study.py --reps 5 --out-dir results/test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from paygap.config import load_config
from paygap.plots import METHOD_LABELS, plot_study, plot_zip, to_percent
from paygap.study import run_study, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--n", type=int)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    cfg = load_config(a.config, reps=a.reps, n=a.n)
    st, me = cfg["study"], cfg["methods"]
    out = Path(a.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    def progress(s, r):
        if (r + 1) % 10 == 0 or r + 1 == st["reps"]:
            print(f"  {s}: {r + 1}/{st['reps']}  ({time.time() - t0:.0f} s)", flush=True)

    runs = run_study(st["scenarios"], st["reps"], st["n"], st["gap"], me["model_params"],
                     me["folds"], st["seed"], progress)
    summary = summarize(runs)
    runs.to_csv(out / "study_runs.csv", index=False)
    summary.to_csv(out / "study_summary.csv", index=False)
    plot_study(runs, out / "figures" / "study.png")
    plot_zip(runs, out / "figures" / "zip_plot.png")
    (out / "run_info.json").write_text(json.dumps(
        {"config": cfg, "seconds": round(time.time() - t0)}, indent=2), encoding="utf-8")

    print(f"\n{'Szenario':<11}{'Methode':<26}{'Mittel':>8}{'Verzerrung':>12}{'RMSE':>7}"
          f"{'Abdeckung':>11}")
    for _, r in summary.iterrows():
        mean = to_percent(r["mean_est"])
        bias = mean - to_percent(r["mean_est"] - r["bias"])
        print(f"{r['scenario']:<11}{METHOD_LABELS[r['method']]:<26}{mean:>7.1f}%"
              f"{bias:>+11.1f}pp{100 * r['rmse']:>6.1f}{r['coverage']:>10.0%}")
    print("\nVerzerrung und RMSE gegenüber der direkten Lücke (RMSE in Log-Punkten x 100).")
    print(f"Ergebnisse gespeichert in {out}/")


if __name__ == "__main__":
    main()
