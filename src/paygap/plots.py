"""Abbildungen: Pastellfarben mit eigenen Markern und Schraffuren je Methode."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METHOD_ORDER = ["raw", "ols", "ml_ob", "dml"]
METHOD_LABELS = {
    "raw": "Unbereinigt",
    "ols": "Lineare Regression",
    "ml_ob": "ML-Oaxaca-Blinder",
    "dml": "Double Machine Learning",
}
METHOD_STYLES = {
    "raw": {"color": "#C9C9C9", "marker": "o", "hatch": ""},
    "ols": {"color": "#A8C8E8", "marker": "s", "hatch": "//"},
    "ml_ob": {"color": "#F6C1A6", "marker": "^", "hatch": ".."},
    "dml": {"color": "#B5E0C5", "marker": "D", "hatch": "xx"},
}
SCENARIO_LABELS = {
    "linear": "Lineare Entgeltstruktur",
    "nonlinear": "Nichtlineare Entgeltstruktur",
    "proxy": "Nichtlinear + seltenere Beförderung",
}


def to_percent(log_gap):
    """Log-Punkte in prozentuale Entgeltdifferenz umrechnen."""
    return 100 * (np.exp(np.asarray(log_gap)) - 1)


def plot_study(runs, path) -> None:
    """Verteilung der Schaetzungen je Szenario und Methode mit den wahren Werten."""
    scenarios = list(dict.fromkeys(runs["scenario"]))
    methods = [m for m in METHOD_ORDER if m in set(runs["method"])]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.6 * len(scenarios), 4.4),
                             sharex=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, s in zip(axes[0], scenarios):
        sub = runs[runs["scenario"] == s]
        for i, m in enumerate(methods):
            st = METHOD_STYLES[m]
            vals = to_percent(sub.loc[sub["method"] == m, "est"])
            bp = ax.boxplot(vals, positions=[i], orientation="horizontal", widths=0.55, patch_artist=True,
                            showfliers=False, medianprops={"color": "#555555"})
            bp["boxes"][0].set(facecolor=st["color"], hatch=st["hatch"], edgecolor="#777777")
            ax.scatter(vals, i + rng.uniform(-0.18, 0.18, len(vals)), s=12, alpha=0.6,
                       marker=st["marker"], color=st["color"], edgecolor="#666666",
                       linewidth=0.4, zorder=3)
        direct = to_percent(sub["true_direct"].iloc[0])
        total = to_percent(sub["true_total"].mean())
        ax.axvline(direct, color="#6A6A9A", ls="--", lw=1.3, label="wahre direkte Lücke")
        if abs(total - direct) > 0.1:
            ax.axvline(total, color="#B07AA1", ls=":", lw=1.6,
                       label="wahre Gesamtlücke (inkl. Beförderung)")
        ax.set_yticks(range(len(methods)), [METHOD_LABELS[m] for m in methods])
        ax.set_title(SCENARIO_LABELS.get(s, s), fontsize=10)
        ax.set_xlabel("geschätzte Entgeltlücke der Frauen (%)")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
    for ax in axes[0][1:]:
        ax.tick_params(labelleft=False)
    handles = {}
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


ZIP_COVERED = "#C5CCD6"      # Intervall enthaelt die wahre Luecke
ZIP_MISSED = "#D9737A"       # Intervall verfehlt sie


def plot_zip(runs, path, methods=("ols", "ml_ob", "dml")) -> None:
    """Zip-Plot (Morris, White & Crowther 2019): alle 95-%-Intervalle je Methode und Szenario.

    Die Intervalle sind von unten nach oben nach |Schaetzung - wahre Luecke| / Standardfehler
    sortiert. Bei korrekter Abdeckung verfehlen nur die obersten 5 % die wahre Luecke.
    Der Standardfehler wird aus der Intervallbreite zurueckgerechnet.
    """
    from matplotlib.lines import Line2D

    scenarios = list(dict.fromkeys(runs["scenario"]))
    methods = [m for m in methods if m in set(runs["method"])]
    fig, axes = plt.subplots(len(methods), len(scenarios),
                             figsize=(4.4 * len(scenarios), 3.3 * len(methods)),
                             sharex=True, sharey=True, squeeze=False)
    for j, s in enumerate(scenarios):
        for i, m in enumerate(methods):
            ax = axes[i, j]
            d = runs[(runs["scenario"] == s) & (runs["method"] == m)]
            se = (d["hi"] - d["lo"]) / (2 * 1.959963984540054)
            order = np.argsort(np.abs((d["est"] - d["true_direct"]) / se).to_numpy(),
                               kind="stable")
            d = d.iloc[order]
            y = 100 * (np.arange(len(d)) + 0.5) / len(d)
            truth = d["true_direct"].to_numpy()
            cov = ((d["lo"] <= truth) & (truth <= d["hi"])).to_numpy()
            lo, hi, est = to_percent(d["lo"]), to_percent(d["hi"]), to_percent(d["est"])
            ax.hlines(y[cov], lo[cov], hi[cov], color=ZIP_COVERED, lw=2.2)
            ax.hlines(y[~cov], lo[~cov], hi[~cov], color=ZIP_MISSED, lw=2.2)
            ax.scatter(est, y, s=6, color="#555555", marker=METHOD_STYLES[m]["marker"],
                       zorder=3)
            ax.axvline(to_percent(truth[0]), color="#6A6A9A", ls="--", lw=1.2)
            ax.axhline(95, color="#888888", ls=":", lw=1.2)
            ax.text(0.98, 0.04, f"Abdeckung {cov.mean():.0%}", transform=ax.transAxes,
                    ha="right", fontsize=9, bbox={"fc": "white", "ec": "none", "alpha": 0.8})
            ax.set_ylim(0, 100)
            ax.grid(axis="x", alpha=0.3)
            if i == 0:
                ax.set_title(SCENARIO_LABELS.get(s, s), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{METHOD_LABELS[m]}\n\nRang nach |Abweichung| / SE (%)")
            if i == len(methods) - 1:
                ax.set_xlabel("95-%-Intervall der Entgeltlücke (%)")
    handles = [
        Line2D([], [], color=ZIP_COVERED, lw=2.2, label="Intervall enthält wahre Lücke"),
        Line2D([], [], color=ZIP_MISSED, lw=2.2, label="Intervall verfehlt wahre Lücke"),
        Line2D([], [], color="#6A6A9A", ls="--", label="wahre direkte Lücke"),
        Line2D([], [], color="#888888", ls=":", label="95-%-Marke"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)
