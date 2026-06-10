from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChannelSchema:
    """Channel identity, units, and sampling metadata."""

    names: tuple[str, ...]
    types: tuple[str, ...]
    units: tuple[str, ...]
    sampling_rate: float
    reference: str = "unknown"
    montage: str | None = None
    bad_channels: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("ChannelSchema.names cannot be empty")
        if len(self.names) != len(self.types) or len(self.names) != len(self.units):
            raise ValueError("ChannelSchema names, types, and units must have equal length")
        if self.sampling_rate <= 0:
            raise ValueError("ChannelSchema.sampling_rate must be positive")

    @property
    def n_channels(self) -> int:
        return len(self.names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "types": list(self.types),
            "units": list(self.units),
            "sampling_rate": self.sampling_rate,
            "reference": self.reference,
            "montage": self.montage,
            "bad_channels": list(self.bad_channels),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Event:
    """A typed event with explicit timing semantics."""

    event_id: str
    event_type: str
    name: str
    onset: float
    duration: float = 0.0
    clock_domain: str = "sample_clock"
    sample_index: int | None = None
    confidence: float = 1.0
    source: str | None = None
    target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "name": self.name,
            "onset": self.onset,
            "duration": self.duration,
            "clock_domain": self.clock_domain,
            "sample_index": self.sample_index,
            "confidence": self.confidence,
            "source": self.source,
            "target": self.target,
            "metadata": self.metadata,
        }


@dataclass
class SignalPacket:
    """Sampled neural or biosignal data."""

    data: np.ndarray
    timestamps: np.ndarray
    channel_schema: ChannelSchema
    modality: str
    events: list[Event] = field(default_factory=list)
    clock_domain: str = "sample_clock"
    source_id: str = "source"
    sequence_id: int = 0
    quality: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("SignalPacket.data must have shape channels x samples")
        if self.data.shape[0] != self.channel_schema.n_channels:
            raise ValueError("SignalPacket channel count does not match ChannelSchema")
        if self.timestamps.ndim != 1 or self.timestamps.shape[0] != self.data.shape[1]:
            raise ValueError("SignalPacket.timestamps must have one timestamp per sample")


@dataclass(frozen=True)
class WindowPacket:
    """A time-windowed view of signal data."""

    window_id: str
    data: np.ndarray
    start_time: float
    end_time: float
    sample_start: int
    sample_end: int
    label: str | None
    events: tuple[Event, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center_time(self) -> float:
        return (self.start_time + self.end_time) / 2.0


@dataclass(frozen=True)
class FeaturePacket:
    """Features extracted from a window."""

    feature_id: str
    features: np.ndarray
    feature_names: tuple[str, ...]
    window_id: str
    label: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntentPacket:
    """Decoder prediction."""

    intent_id: str
    intent: str
    confidence: float
    posterior: dict[str, float]
    latency_ms: float
    window_id: str
    decoder_id: str
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

