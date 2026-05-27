"""Tests for PipelineConfig — behavior through public interface only."""

import math
import pytest


# ── Cycle 1: default values match reference experiment ───────────────────

def test_defaults_match_reference_experiment():
    from pyorica.config import PipelineConfig
    cfg = PipelineConfig()
    assert cfg.asr_backend == "asrpy"
    assert cfg.asr_cutoff == 20.0
    assert cfg.icalabel_threshold == 0.7
    assert cfg.asr_calibration_seconds == 120.0


# ── Cycle 2: YAML round-trip preserves all values including inf ───────────

def test_yaml_roundtrip(tmp_path):
    from pyorica.config import PipelineConfig
    cfg = PipelineConfig(asr_cutoff=15.0, icalabel_threshold=0.5)
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = PipelineConfig.from_yaml(path)
    assert loaded == cfg


def test_yaml_roundtrip_inf(tmp_path):
    from pyorica.config import PipelineConfig
    cfg = PipelineConfig(orica_tau_const=float("inf"))
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = PipelineConfig.from_yaml(path)
    assert math.isinf(loaded.orica_tau_const)


# ── Cycle 3: EEGPipeline accepts PipelineConfig ───────────────────────────

def test_pipeline_accepts_config():
    import numpy as np
    from pyorica.config import PipelineConfig
    from pyorica.pipeline.pipeline import EEGPipeline
    cfg = PipelineConfig(asr_backend="meegkit", iir_l_freq=1.0, iir_h_freq=50.0)
    p = EEGPipeline(n_channels=8, sfreq=256.0, config=cfg)
    chunk = np.random.default_rng(0).standard_normal((8, 64))
    out = p.process(chunk)
    assert out.shape == (8, 64)


def test_pipeline_config_overrides_kwargs():
    """config takes precedence over individual kwargs when both are provided."""
    from pyorica.config import PipelineConfig
    from pyorica.pipeline.pipeline import EEGPipeline
    cfg = PipelineConfig(asr_backend="meegkit", asr_cutoff=10.0)
    # pass conflicting kwarg — config should win
    p = EEGPipeline(n_channels=8, sfreq=256.0, asr_backend="asrpy", config=cfg)
    assert p._asr._backend == "meegkit"
    assert p._asr._cutoff == 10.0
