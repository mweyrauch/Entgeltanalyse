"""Konfiguration laden."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path, **overrides) -> dict:
    """Liest eine YAML-Datei; gesetzte Overrides (nicht None) ersetzen Werte unter `study`."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key, value in overrides.items():
        if value is not None:
            cfg["study"][key] = value
    return cfg
