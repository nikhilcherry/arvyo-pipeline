#!/usr/bin/env python
# FROZEN — do not edit during the finale. Changes require re-testing on a fresh environment.
"""The single canonical environment setup cell for the August finale notebook.

Paste this as the first cell of the finale Colab/Kaggle notebook, or run it
directly on any machine: `python docs/finale_setup_cell.py`.

What it does, in order:
  1. Installs team Arvyo's five command-line tools, exclusively via the
     git-install form. Bare `pip install foldr` (or trackr/batchr/peekr/fitr)
     is forbidden workspace-wide: `foldr` is squatted on PyPI by an unrelated
     package, and none of these tools are published there.
  2. Installs the pipeline's scientific dependencies at the exact specifiers
     already declared in arvyo-pipeline/requirements.txt / pyproject.toml —
     nothing here invents a version pin that isn't already in the repo.
  3. Runs a self-check: imports every package, prints its version, and
     hard-fails with a clear message naming exactly what's missing.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

TOOLS = ["trackr", "batchr", "peekr", "foldr", "fitr"]

# Mirrors requirements.txt/pyproject.toml's `dependencies` list exactly, for
# the subset that the finale notebook's analysis-by-synthesis walkthrough
# actually touches (search, forward models, MCMC fitting). `astropy` is not
# declared in requirements.txt but is imported directly by
# third_party/nigraha_views/preprocess.py and arvyo/vetting/vet.py; there is
# no existing pin for it to read, so — like most entries below — it installs
# unpinned rather than inventing one.
SCI_DEPS = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "lightkurve>=2.4",
    "wotan",
    "transitleastsquares",
    "batman-package",
    "emcee",
    "astropy",
]

# pip spec name -> import name, where they differ.
IMPORT_NAMES = {
    "batman-package": "batman",
}


def _pip_install(spec: str) -> None:
    print(f"[setup] installing {spec} ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec], check=True)


def install_tools() -> None:
    for name in TOOLS:
        _pip_install(f"git+https://github.com/nikhilcherry/{name}")


def install_sci_deps() -> None:
    for spec in SCI_DEPS:
        _pip_install(spec)


def self_check() -> None:
    modules = {name: name for name in TOOLS}
    for spec in SCI_DEPS:
        base = spec.split(">=")[0].split("==")[0]
        modules[base] = IMPORT_NAMES.get(base, base)

    missing = []
    print("\n[setup] self-check:")
    for label, import_name in sorted(modules.items()):
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "ok")
            print(f"  {label:22s} -> {version}")
        except ImportError as exc:
            missing.append(label)
            print(f"  {label:22s} -> MISSING ({exc})")

    if missing:
        print(f"\n[setup] FAILED: missing package(s): {', '.join(missing)}", file=sys.stderr)
        print("[setup] Re-run this cell; if it persists, check network/auth "
              "access to github.com.", file=sys.stderr)
        sys.exit(1)
    print("\n[setup] all packages present.")


def main() -> None:
    install_tools()
    install_sci_deps()
    self_check()


if __name__ == "__main__":
    main()
