from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class UniformRange:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class DomainRandomization:
    """Sample source override dictionaries for synthetic robustness sweeps."""

    snr_db: UniformRange | None = None
    electrode_shift_mm: UniformRange | None = None
    blink_rate_per_min: UniformRange | None = None
    muscle_noise: UniformRange | None = None
    channel_dropout_probability: UniformRange | None = None
    amplitude_drift: UniformRange | None = None
    spectral_drift_hz: UniformRange | None = None
    spatial_covariance_drift: UniformRange | None = None
    marker_jitter_ms: UniformRange | None = None
    feedback_delay_ms: UniformRange | None = None
    attention: UniformRange | None = None
    fatigue_rate: UniformRange | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def sample_source_overrides(self, seed: int | None = None) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        overrides: dict[str, Any] = {"subject": {}, "session": {}}
        if self.snr_db is not None:
            overrides["snr_db"] = self.snr_db.sample(rng)
        for key in ("attention", "fatigue_rate"):
            value = getattr(self, key)
            if value is not None:
                overrides["subject"][key] = value.sample(rng)
        for key in (
            "electrode_shift_mm",
            "blink_rate_per_min",
            "muscle_noise",
            "channel_dropout_probability",
            "amplitude_drift",
            "spectral_drift_hz",
            "spatial_covariance_drift",
            "marker_jitter_ms",
            "feedback_delay_ms",
        ):
            value = getattr(self, key)
            if value is not None:
                overrides["session"][key] = value.sample(rng)
        if not overrides["subject"]:
            overrides.pop("subject")
        if not overrides["session"]:
            overrides.pop("session")
        return overrides

