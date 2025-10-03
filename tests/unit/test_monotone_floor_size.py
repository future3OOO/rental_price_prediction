import os
from importlib import reload
import model_training as mt


def test_mono_from_env_defaults_include_floor_size(monkeypatch):
    # Ensure ENFORCE_MONOTONE default kicks in
    monkeypatch.setenv("ENFORCE_MONOTONE", "1")
    # Clear explicit env knobs
    monkeypatch.delenv("MONO_INC", raising=False)
    monkeypatch.delenv("MONO_DEC", raising=False)
    # Reload to pick up module-level flags
    reload(mt)
    cols = [
        "Capital Value",
        "Floor Size (sqm)",
        "Land Size (sqm)",
        "Bath",
        "Car",
    ]
    mono = mt._mono_from_env(cols)
    assert mono is not None
    # Expect +1 for Bath, Car, and Floor Size (sqm)
    mapping = dict(zip(cols, mono))
    assert mapping["Bath"] == 1
    assert mapping["Car"] == 1
    assert mapping["Floor Size (sqm)"] == 1


def test_mono_from_env_respects_mono_inc(monkeypatch):
    monkeypatch.setenv("MONO_INC", "Floor Size (sqm)")
    monkeypatch.delenv("MONO_DEC", raising=False)
    cols = ["X", "Floor Size (sqm)", "Y"]
    mono = mt._mono_from_env(cols)
    assert mono == [0, 1, 0]

