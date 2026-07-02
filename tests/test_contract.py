from pathlib import Path

import numpy as np
import pytest

from arvyo.contract import ContractError, load_sample, validate_dataset

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", [
    "fixture_planet.npz",
    "fixture_eb.npz",
    "fixture_starspot.npz",
    "fixture_null.npz",
    "fixture_unknown.npz",
])
def test_fixtures_validate(name):
    sample = load_sample(FIXTURES / name)
    assert sample["label"] in {"planet", "eb", "starspot", "null", "unknown"}
    assert sample["time"].shape == sample["flux"].shape == sample["flux_err"].shape


def test_broken_fixture_missing_flux_err(tmp_path):
    path = tmp_path / "broken.npz"
    np.savez(path, time=np.linspace(0, 1, 10), flux=np.ones(10),
             tic_id=1, label="planet", sector=1)

    with pytest.raises(ContractError) as excinfo:
        load_sample(path)

    assert "broken.npz" in str(excinfo.value)
    assert "flux_err" in str(excinfo.value)


def test_validate_dataset_on_fixtures(tmp_path):
    for f in FIXTURES.glob("*.npz"):
        sample = load_sample(f)
        label_dir = tmp_path / sample["label"]
        label_dir.mkdir(exist_ok=True)
        (label_dir / f.name).write_bytes(f.read_bytes())

    report = validate_dataset(tmp_path)
    assert sum(report["counts"].values()) == 5
    assert report["invalid"] == []
