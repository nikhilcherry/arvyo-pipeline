"""Config for the foldr -> fitr glue pipeline (arvyo/worker.py, batch_worker.py).

Loaded from the `pipeline:` section of configs/default.yaml. Resolution is
package-relative, not cwd-relative, so `python -m arvyo.run` behaves the
same regardless of the caller's working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@dataclass(frozen=True)
class PipelineConfig:
    sde_min: float = 7.0
    snr_min: float = 7.0
    foldr_timeout_s: float = 300
    fitr_timeout_s: float = 600
    results_dir: str = "results"
    # Off by default: localizr needs live network access (MAST + Gaia) for
    # a real target pixel file, which most test/CI/offline environments
    # don't have. Opt in via configs/default.yaml's `pipeline:` section or
    # `arvyo.run one --centroid-vet`. See worker.py's _run_centroid_vetting.
    centroid_vetting_enabled: bool = False
    localizr_timeout_s: float = 120

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PipelineConfig":
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        section = raw.get("pipeline", {})
        defaults = cls()
        return cls(
            sde_min=float(section.get("sde_min", defaults.sde_min)),
            snr_min=float(section.get("snr_min", defaults.snr_min)),
            foldr_timeout_s=float(section.get("foldr_timeout_s", defaults.foldr_timeout_s)),
            fitr_timeout_s=float(section.get("fitr_timeout_s", defaults.fitr_timeout_s)),
            results_dir=str(section.get("results_dir", defaults.results_dir)),
            centroid_vetting_enabled=bool(
                section.get("centroid_vetting_enabled", defaults.centroid_vetting_enabled)
            ),
            localizr_timeout_s=float(
                section.get("localizr_timeout_s", defaults.localizr_timeout_s)
            ),
        )

    def as_dict(self) -> dict:
        return {
            "sde_min": self.sde_min,
            "snr_min": self.snr_min,
            "foldr_timeout_s": self.foldr_timeout_s,
            "fitr_timeout_s": self.fitr_timeout_s,
            "results_dir": self.results_dir,
            "centroid_vetting_enabled": self.centroid_vetting_enabled,
            "localizr_timeout_s": self.localizr_timeout_s,
        }
