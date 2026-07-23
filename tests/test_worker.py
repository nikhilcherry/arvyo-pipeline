"""Unit tests for arvyo.worker's centroid-vetting decision logic
(_run_centroid_vetting) -- pure logic, no real localizr/network call, via
monkeypatching arvyo.worker._invoke_tool.
"""
from __future__ import annotations

import pytest

from arvyo import worker
from arvyo.pipeline_config import PipelineConfig

FULL_PERIOD_SEARCH = {
    "engine": "tls", "period": 3.14, "t0": 1500.0,
    "duration_hours": 4.2, "depth_ppm": 800.0,
    "snr": 12.0, "sde": 15.0, "passed_gate": True,
}
CATALOG_PERIOD_SEARCH = {
    "engine": "catalog", "period": 3.14, "t0": 1500.0,
    "duration_hours": None, "depth_ppm": None,
    "snr": None, "sde": None, "passed_gate": True,
}


def test_disabled_returns_none():
    config = PipelineConfig()
    result = worker._run_centroid_vetting(
        {"tic_id": 123}, FULL_PERIOD_SEARCH, "clear", "planet", None, config, centroid_vet=False
    )
    assert result is None


def test_no_significant_signal_is_skipped_not_invoked(monkeypatch):
    called = []
    monkeypatch.setattr(worker, "_invoke_tool", lambda *a, **k: called.append(1))
    config = PipelineConfig()
    result = worker._run_centroid_vetting(
        {"tic_id": 123}, FULL_PERIOD_SEARCH, "no_significant_signal", None, None, config,
        centroid_vet=True,
    )
    assert result == {"ran": False, "skipped_reason": "verdict/winner is not a planet/blend case"}
    assert called == []


def test_clear_starspot_winner_is_skipped_not_invoked(monkeypatch):
    called = []
    monkeypatch.setattr(worker, "_invoke_tool", lambda *a, **k: called.append(1))
    config = PipelineConfig()
    result = worker._run_centroid_vetting(
        {"tic_id": 123}, FULL_PERIOD_SEARCH, "clear", "starspot", None, config, centroid_vet=True
    )
    assert result["ran"] is False
    assert called == []


def test_clear_planet_winner_invokes_localizr(monkeypatch):
    captured = {}

    def fake_invoke(name, args, timeout_s):
        captured["name"] = name
        captured["args"] = args
        captured["timeout_s"] = timeout_s
        return 0, {"verdict": "on_target", "centroid_offset_sigma": 0.8}, "", 3.5, None

    monkeypatch.setattr(worker, "_invoke_tool", fake_invoke)
    config = PipelineConfig(localizr_timeout_s=99)
    result = worker._run_centroid_vetting(
        {"tic_id": 4281068}, FULL_PERIOD_SEARCH, "clear", "planet", None, config,
        centroid_vet=True,
    )
    assert captured["name"] == "localizr"
    assert captured["timeout_s"] == 99
    assert "--tic-id" in captured["args"]
    assert "4281068" in captured["args"]
    assert "--period" in captured["args"] and "3.14" in captured["args"]
    assert result["ran"] is True
    assert result["verdict"] == "on_target"
    assert result["runtime_s"] == 3.5


def test_ambiguous_planet_blend_tie_invokes_localizr(monkeypatch):
    monkeypatch.setattr(
        worker, "_invoke_tool",
        lambda *a, **k: (0, {"verdict": "inconclusive"}, "", 1.0, None),
    )
    config = PipelineConfig()
    model_fit = {"tied_models": ["planet", "blend"]}
    result = worker._run_centroid_vetting(
        {"tic_id": 123}, FULL_PERIOD_SEARCH, "ambiguous", None, model_fit, config,
        centroid_vet=True,
    )
    assert result["ran"] is True


def test_ambiguous_without_blend_in_tie_is_skipped(monkeypatch):
    called = []
    monkeypatch.setattr(worker, "_invoke_tool", lambda *a, **k: called.append(1))
    config = PipelineConfig()
    model_fit = {"tied_models": ["planet", "eb"]}
    result = worker._run_centroid_vetting(
        {"tic_id": 123}, FULL_PERIOD_SEARCH, "ambiguous", None, model_fit, config,
        centroid_vet=True,
    )
    assert result["ran"] is False
    assert called == []


def test_missing_tic_id_is_skipped_not_invoked(monkeypatch):
    called = []
    monkeypatch.setattr(worker, "_invoke_tool", lambda *a, **k: called.append(1))
    config = PipelineConfig()
    result = worker._run_centroid_vetting(
        {"tic_id": None}, FULL_PERIOD_SEARCH, "clear", "planet", None, config,
        centroid_vet=True,
    )
    assert result == {"ran": False, "skipped_reason": "no tic_id/kic_id in .npz metadata"}
    assert called == []


def test_catalog_period_path_is_skipped_no_duration(monkeypatch):
    called = []
    monkeypatch.setattr(worker, "_invoke_tool", lambda *a, **k: called.append(1))
    config = PipelineConfig()
    result = worker._run_centroid_vetting(
        {"tic_id": 123}, CATALOG_PERIOD_SEARCH, "clear", "planet", None, config,
        centroid_vet=True,
    )
    assert result["ran"] is False
    assert "duration_hours" in result["skipped_reason"]
    assert called == []


def test_kepler_mission_uses_kic_id_flag(monkeypatch):
    captured = {}

    def fake_invoke(name, args, timeout_s):
        captured["args"] = args
        return 0, {"verdict": "on_target"}, "", 1.0, None

    monkeypatch.setattr(worker, "_invoke_tool", fake_invoke)
    config = PipelineConfig()
    worker._run_centroid_vetting(
        {"tic_id": 6974867, "mission": "kepler"}, FULL_PERIOD_SEARCH, "clear", "blend", None,
        config, centroid_vet=True,
    )
    assert "--kic-id" in captured["args"]
    assert "--tic-id" not in captured["args"]


def test_localizr_failure_reports_skipped_not_raises(monkeypatch):
    monkeypatch.setattr(
        worker, "_invoke_tool",
        lambda *a, **k: (None, None, "", 0.1, "'localizr' executable not found"),
    )
    config = PipelineConfig()
    result = worker._run_centroid_vetting(
        {"tic_id": 123}, FULL_PERIOD_SEARCH, "clear", "planet", None, config,
        centroid_vet=True,
    )
    assert result["ran"] is False
    assert "not found" in result["skipped_reason"]
