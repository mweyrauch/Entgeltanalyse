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
