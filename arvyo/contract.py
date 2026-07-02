"""The frozen arvyo-data <-> arvyo-pipeline .npz schema: loader + validator.

This is the ONLY interface between the two repos. Any change to this schema
requires bumping SCHEMA_VERSION and updating BOTH repos' READMEs.
"""

from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0"

REQUIRED_ARRAYS = ["time", "flux", "flux_err"]
OPTIONAL_ARRAYS = ["flux_raw"]          # present for starspot/null classes
REQUIRED_META = ["tic_id", "label", "sector"]
OPTIONAL_META = ["period_days", "epoch_btjd", "crowdsap", "mission",
                 "augmented", "injection_params"]
LABELS = ["planet", "eb", "blend", "starspot", "null", "unknown"]


class ContractError(ValueError):
    """Raised when a .npz sample violates the data contract."""


def _scalar(value):
    arr = np.asarray(value)
    return arr.item() if arr.shape == () else arr


def load_sample(path):
    """Load one .npz sample and validate it against the schema.

    Returns a dict of the required/optional arrays and metadata.
    Raises ContractError naming the file and the violated rule.
    """
    path = Path(path)
    if not path.exists():
        raise ContractError(f"{path}: file does not exist")
    if path.stat().st_size == 0:
        raise ContractError(f"{path}: file is zero bytes")

    try:
        npz = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise ContractError(f"{path}: could not load npz ({exc})") from exc

    keys = set(npz.files)

    missing_arrays = [k for k in REQUIRED_ARRAYS if k not in keys]
    if missing_arrays:
        raise ContractError(f"{path}: missing required array(s) {missing_arrays}")

    missing_meta = [k for k in REQUIRED_META if k not in keys]
    if missing_meta:
        raise ContractError(f"{path}: missing required meta field(s) {missing_meta}")

    time = np.asarray(npz["time"], dtype=np.float64)
    flux = np.asarray(npz["flux"], dtype=np.float64)
    flux_err = np.asarray(npz["flux_err"], dtype=np.float64)

    if time.ndim != 1:
        raise ContractError(f"{path}: 'time' must be 1D, got shape {time.shape}")
    if flux.shape != time.shape:
        raise ContractError(f"{path}: 'flux' shape {flux.shape} != 'time' shape {time.shape}")
    if flux_err.shape != time.shape:
        raise ContractError(f"{path}: 'flux_err' shape {flux_err.shape} != 'time' shape {time.shape}")

    sample = {"time": time, "flux": flux, "flux_err": flux_err}

    if "flux_raw" in keys:
        flux_raw = np.asarray(npz["flux_raw"], dtype=np.float64)
        if flux_raw.shape != time.shape:
            raise ContractError(f"{path}: 'flux_raw' shape {flux_raw.shape} != 'time' shape {time.shape}")
        sample["flux_raw"] = flux_raw

    label = str(_scalar(npz["label"]))
    if label not in LABELS:
        raise ContractError(f"{path}: label {label!r} not in {LABELS}")
    sample["label"] = label

    sample["tic_id"] = _scalar(npz["tic_id"])
    sample["sector"] = _scalar(npz["sector"])

    for key in OPTIONAL_META:
        if key in keys:
            sample[key] = _scalar(npz[key])

    return sample


def validate_dataset(root_dir):
    """Walk {root}/{label}/*.npz, returning per-label counts + invalid files.

    Zero-byte and unloadable files are skipped gracefully (arvyo-data may
    still be writing them).
    """
    root = Path(root_dir)
    report = {"root": str(root), "counts": {}, "invalid": [], "skipped": []}

    if not root.exists():
        report["error"] = f"{root} does not exist"
        return report

    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = label_dir.name
        count = 0
        for npz_path in sorted(label_dir.glob("*.npz")):
            if npz_path.stat().st_size == 0:
                report["skipped"].append(str(npz_path))
                continue
            try:
                load_sample(npz_path)
                count += 1
            except ContractError as exc:
                report["invalid"].append({"path": str(npz_path), "error": str(exc)})
            except Exception:
                report["skipped"].append(str(npz_path))
        report["counts"][label] = count

    return report


def format_report(report):
    lines = [f"Data contract report for {report['root']} (schema {SCHEMA_VERSION})"]
    if "error" in report:
        lines.append(f"  ERROR: {report['error']}")
        return "\n".join(lines)
    for label, count in report["counts"].items():
        lines.append(f"  {label}: {count} valid")
    if report["skipped"]:
        lines.append(f"  skipped (zero-byte/unloadable): {len(report['skipped'])}")
    if report["invalid"]:
        lines.append(f"  INVALID: {len(report['invalid'])}")
        for entry in report["invalid"]:
            lines.append(f"    - {entry['path']}: {entry['error']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m arvyo.contract /path/to/processed", file=sys.stderr)
        sys.exit(1)

    print(format_report(validate_dataset(sys.argv[1])))
