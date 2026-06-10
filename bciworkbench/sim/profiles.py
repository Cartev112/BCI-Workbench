from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubjectProfile:
    """Synthetic subject parameters that should persist across sessions."""

    alpha_peak_hz: float = 10.0
    beta_peak_hz: float = 20.0
    motor_imagery_vividness: float = 1.0
    attention: float = 0.85
    fatigue_rate: float = 0.0

    @classmethod
    def from_dict(cls, values: dict | None) -> "SubjectProfile":
        return cls(**(values or {}))

    def to_dict(self) -> dict[str, float]:
        return {
            "alpha_peak_hz": self.alpha_peak_hz,
            "beta_peak_hz": self.beta_peak_hz,
            "motor_imagery_vividness": self.motor_imagery_vividness,
            "attention": self.attention,
            "fatigue_rate": self.fatigue_rate,
        }


@dataclass(frozen=True)
class SessionProfile:
    """Synthetic session parameters that vary across recordings."""

    amplitude_drift: float = 0.05
    electrode_shift_mm: float = 0.0
    blink_rate_per_min: float = 0.0
    blink_amplitude_uv: float = 90.0
    muscle_noise: float = 0.0
    channel_dropout_probability: float = 0.0
    marker_jitter_ms: float = 0.0

    @classmethod
    def from_dict(cls, values: dict | None) -> "SessionProfile":
        return cls(**(values or {}))

    def to_dict(self) -> dict[str, float]:
        return {
            "amplitude_drift": self.amplitude_drift,
            "electrode_shift_mm": self.electrode_shift_mm,
            "blink_rate_per_min": self.blink_rate_per_min,
            "blink_amplitude_uv": self.blink_amplitude_uv,
            "muscle_noise": self.muscle_noise,
            "channel_dropout_probability": self.channel_dropout_probability,
            "marker_jitter_ms": self.marker_jitter_ms,
        }

