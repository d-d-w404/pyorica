"""PipelineConfig: all parameters needed to reproduce a pipeline run."""

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Union


@dataclass
class PipelineConfig:
    # IIR bandpass
    iir_l_freq: float = 1.0
    iir_h_freq: float = 50.0
    iir_order: int = 4
    # ASR
    asr_backend: str = "asrpy"
    asr_cutoff: float = 20.0
    asr_calibration_seconds: float = 120.0
    # ORICA
    orica_ff_profile: str = "cooling"
    orica_block_size_white: int = 8
    orica_block_size_ica: int = 8
    orica_lambda_0: float = 0.995
    orica_gamma: float = 0.6
    orica_num_subgaussian: int = 0
    orica_tau_const: float = float("inf")
    # ICLabel
    icalabel_threshold: float = 0.7

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Serialize config to a YAML file."""
        import yaml
        d = asdict(self)
        # represent inf as .inf so YAML round-trips cleanly
        if math.isinf(d.get("orica_tau_const", 0)):
            d["orica_tau_const"] = ".inf"
        with open(path, "w") as f:
            yaml.safe_dump(d, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Load config from a YAML file."""
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        # restore .inf → float("inf")
        if d.get("orica_tau_const") == ".inf":
            d["orica_tau_const"] = float("inf")
        return cls(**d)
