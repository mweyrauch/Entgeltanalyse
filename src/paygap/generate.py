"""Synthetische Personaldaten mit bekannter Entgeltluecke.

Zielgroesse ist das logarithmierte Stundenentgelt. Die eingebaute direkte Luecke `gap`
wirkt bei gleichen Merkmalen auf das Entgelt von Frauen (in Log-Punkten, -0.05 ~ -5 %).

Szenarien:
- linear:    Entgelt haengt linear und additiv von den Merkmalen ab.
- nonlinear: Erfahrung wirkt saettigend, Tarifstufen springen, Standortzuschlaege haengen
             von der Jobfamilie ab, der Teilzeitabschlag waechst mit der Entgeltstufe.
- proxy:     wie nonlinear, zusaetzlich werden Frauen bei gleicher Qualifikation seltener
             befoerdert. Wer die Entgeltstufe herausrechnet, sieht diesen Teil nicht.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIOS = ("linear", "nonlinear", "proxy")
JOBS = ["Verwaltung", "Vertrieb", "IT", "Produktion", "Logistik", "Forschung"]
LOCATIONS = ["Nord", "Sued", "West", "Ost"]

# Verteilung der Jobfamilien nach Geschlecht (Frauen und Maenner arbeiten in
# unterschiedlichen Bereichen; das ist ein erklaerter, kein unerklaerter Unterschied)
P_JOB_M = np.array([0.10, 0.18, 0.25, 0.25, 0.17, 0.05])
P_JOB_F = np.array([0.30, 0.22, 0.12, 0.08, 0.08, 0.20])
JOB_BASE = np.array([0.00, 0.05, 0.25, -0.05, -0.10, 0.20])
JOB_LEVEL = np.array([0.0, 0.2, 0.6, -0.3, -0.5, 0.5])       # Einfluss auf die Entgeltstufe
LOC_BASE = np.array([0.03, 0.06, 0.00, -0.08])
# Standortzuschlag je Jobfamilie (nur nichtlinear): IT im Sueden besonders gut bezahlt usw.
LOC_JOB = np.array([
    [0.00, 0.02, 0.06, 0.00, -0.02, 0.04],
    [0.02, 0.04, 0.12, 0.02, 0.00, 0.08],
    [0.00, 0.00, 0.03, 0.03, 0.02, 0.00],
    [-0.02, -0.03, -0.06, 0.02, 0.01, -0.04],
])
LEVEL_CUTS = np.array([1.0, 2.0, 3.0, 4.0])                # Grenzen fuer die Stufen 1..5
PROMOTION_PENALTY = 0.6     # proxy: geringere Befoerderungsneigung von Frauen (latente Einheit)


def _levels(latent: np.ndarray) -> np.ndarray:
    return np.digitize(latent, LEVEL_CUTS) + 1


def _log_wage(scenario, job, loc, exp, level, part, rng_noise) -> np.ndarray:
    base = 3.0 + JOB_BASE[job]
    if scenario == "linear":
        return (base + 0.10 * level + 0.010 * exp + LOC_BASE[loc] - 0.06 * part
                + rng_noise)
    # nonlinear und proxy
    return (base + 0.10 * level + 0.08 * (level >= 4)                 # Stufensprung
            + 0.30 * (1 - np.exp(-exp / 7))                          # saettigende Erfahrung
            + LOC_BASE[loc] + LOC_JOB[loc, job]                      # Standort x Jobfamilie
            - part * (0.02 + 0.04 * (level - 1))                     # Teilzeitabschlag je Stufe
            + rng_noise)


def generate(n: int, scenario: str, gap: float = -0.05, seed: int = 0):
    """Erzeugt einen Datensatz und die wahren Lueckenwerte.

    Rueckgabe: (DataFrame, truth) mit truth = {"direct": ..., "total": ...}.
    direct: Luecke bei gleichen Merkmalen einschliesslich Entgeltstufe (= gap).
    total:  Luecke einschliesslich des Umwegs ueber seltenere Befoerderung (nur proxy
            verschieden), gemittelt ueber die Frauen.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unbekanntes Szenario: {scenario}")
    rng = np.random.default_rng(seed)
    female = rng.random(n) < 0.45
    u_job = rng.random(n)
    job = np.where(female,
                   np.searchsorted(np.cumsum(P_JOB_F), u_job * P_JOB_F.sum()),
                   np.searchsorted(np.cumsum(P_JOB_M), u_job * P_JOB_M.sum()))
    job = np.minimum(job, len(JOBS) - 1)
    loc = rng.integers(0, len(LOCATIONS), n)
    age = rng.uniform(22, 64, n)
    breaks = rng.poisson(np.where(female, 2.0, 0.4))              # Erwerbsunterbrechungen
    exp = np.clip(age - 22 - breaks - rng.uniform(0, 3, n), 0, None)
    part = (rng.random(n) < np.where(female, 0.40, 0.08)).astype(int)
    level_noise = rng.normal(0, 0.8, n)
    latent = 0.5 + 0.09 * exp + JOB_LEVEL[job] - 0.4 * part + level_noise
    level_cf = _levels(latent)                                   # Stufe ohne Benachteiligung
    if scenario == "proxy":
        level = _levels(latent - PROMOTION_PENALTY * female)
    else:
        level = level_cf
    noise = rng.normal(0, 0.12, n)

    lw = _log_wage(scenario, job, loc, exp, level, part, noise) + gap * female
    lw_cf = _log_wage(scenario, job, loc, exp, level_cf, part, noise)  # als Mann behandelt

    df = pd.DataFrame({
        "female": female.astype(int),
        "job": pd.Categorical.from_codes(job, JOBS),
        "location": pd.Categorical.from_codes(loc, LOCATIONS),
        "age": age.round(1),
        "experience": exp.round(1),
        "parttime": part,
        "level": level,
        "log_wage": lw,
    })
    df["hourly_wage"] = np.exp(df["log_wage"]).round(2)
    truth = {"direct": gap, "total": float(np.mean((lw - lw_cf)[female]))}
    return df, truth
