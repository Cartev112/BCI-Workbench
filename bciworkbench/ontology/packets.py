from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bciworkbench.ontology.validation import (
    require_channel_types,
    require_clock_domain,
    require_modality,
    require_monotonic_timestamps,
)


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
        if len(set(self.names)) != len(self.names):
            raise ValueError("ChannelSchema.names must be unique")
        if self.sampling_rate <= 0:
            raise ValueError("ChannelSchema.sampling_rate must be positive")
        require_channel_types(self.types)
        missing_bad_channels = sorted(set(self.bad_channels) - set(self.names))
        if missing_bad_channels:
            raise ValueError(f"bad channel(s) not present in names: {', '.join(missing_bad_channels)}")

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

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("Event.event_id cannot be empty")
        if not self.event_type:
            raise ValueError("Event.event_type cannot be empty")
        if not self.name:
            raise ValueError("Event.name cannot be empty")
        if self.onset < 0:
            raise ValueError("Event.onset must be non-negative")
        if self.duration < 0:
            raise ValueError("Event.duration must be non-negative")
        require_clock_domain(self.clock_domain)
        if self.sample_index is not None and self.sample_index < 0:
            raise ValueError("Event.sample_index must be non-negative when provided")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Event.confidence must be between 0 and 1")

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
        require_clock_domain(self.clock_domain)
        require_modality(self.modality)
        require_monotonic_timestamps(self.timestamps)

    def to_dict(self, include_data: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "shape": list(self.data.shape),
            "timestamps_start": float(self.timestamps[0]) if self.timestamps.size else None,
            "timestamps_end": float(self.timestamps[-1]) if self.timestamps.size else None,
            "clock_domain": self.clock_domain,
            "channel_schema": self.channel_schema.to_dict(),
            "modality": self.modality,
            "events": [event.to_dict() for event in self.events],
            "source_id": self.source_id,
            "sequence_id": self.sequence_id,
            "quality": self.quality,
            "metadata": self.metadata,
        }
        if include_data:
            payload["data"] = self.data.tolist()
            payload["timestamps"] = self.timestamps.tolist()
        return payload


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

    def __post_init__(self) -> None:
        if not self.window_id:
            raise ValueError("WindowPacket.window_id cannot be empty")
        if self.start_time < 0 or self.end_time < 0:
            raise ValueError("WindowPacket times must be non-negative")
        if self.end_time <= self.start_time:
            raise ValueError("WindowPacket.end_time must be greater than start_time")
        if self.sample_start < 0 or self.sample_end <= self.sample_start:
            raise ValueError("WindowPacket sample range is invalid")

    def to_dict(self, include_data: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "window_id": self.window_id,
            "shape": list(self.data.shape),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "center_time": self.center_time,
            "sample_start": self.sample_start,
            "sample_end": self.sample_end,
            "label": self.label,
            "events": [event.to_dict() for event in self.events],
            "metadata": self.metadata,
        }
        if include_data:
            payload["data"] = self.data.tolist()
        return payload


@dataclass(frozen=True)
class FeaturePacket:
    """Features extracted from a window."""

    feature_id: str
    features: np.ndarray
    feature_names: tuple[str, ...]
    window_id: str
    label: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_id:
            raise ValueError("FeaturePacket.feature_id cannot be empty")
        if self.features.ndim != 1:
            raise ValueError("FeaturePacket.features must be one-dimensional")
        if len(self.feature_names) != self.features.shape[0]:
            raise ValueError("FeaturePacket.feature_names must match feature count")

    def to_dict(self, include_features: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "feature_id": self.feature_id,
            "feature_count": int(self.features.shape[0]),
            "feature_names": list(self.feature_names),
            "window_id": self.window_id,
            "label": self.label,
            "metadata": self.metadata,
        }
        if include_features:
            payload["features"] = self.features.tolist()
        return payload


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

    def __post_init__(self) -> None:
        if not self.intent_id:
            raise ValueError("IntentPacket.intent_id cannot be empty")
        if not self.intent:
            raise ValueError("IntentPacket.intent cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("IntentPacket.confidence must be between 0 and 1")
        if self.latency_ms < 0:
            raise ValueError("IntentPacket.latency_ms must be non-negative")
        if not self.window_id:
            raise ValueError("IntentPacket.window_id cannot be empty")
        if not self.decoder_id:
            raise ValueError("IntentPacket.decoder_id cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "intent": self.intent,
            "confidence": self.confidence,
            "posterior": self.posterior,
            "latency_ms": self.latency_ms,
            "window_id": self.window_id,
            "decoder_id": self.decoder_id,
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TaskStatePacket:
    """State of a closed-loop task at a point in time."""

    task_id: str
    state: dict[str, Any]
    observation: dict[str, Any]
    target: str | None
    reward: float | None
    done: bool
    success: bool
    events: tuple[Event, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("TaskStatePacket.task_id cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "state": self.state,
            "observation": self.observation,
            "target": self.target,
            "reward": self.reward,
            "done": self.done,
            "success": self.success,
            "events": [event.to_dict() for event in self.events],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class FeedbackPacket:
    """Feedback delivered to a human, simulated user, or environment."""

    action: str
    rendered_at: float
    clock_domain: str
    reward: float | None
    delay_ms: float
    task_state: TaskStatePacket | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("FeedbackPacket.action cannot be empty")
        if self.rendered_at < 0:
            raise ValueError("FeedbackPacket.rendered_at must be non-negative")
        if self.delay_ms < 0:
            raise ValueError("FeedbackPacket.delay_ms must be non-negative")
        require_clock_domain(self.clock_domain)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "rendered_at": self.rendered_at,
            "clock_domain": self.clock_domain,
            "reward": self.reward,
            "delay_ms": self.delay_ms,
            "task_state": self.task_state.to_dict() if self.task_state else None,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AdaptationPacket:
    """Record of a decoder, policy, threshold, calibration, or user-model update."""

    adapter_id: str
    update_type: str
    input_window_ids: tuple[str, ...]
    labels: tuple[str, ...] = ()
    confidence_gate: float | None = None
    parameters_changed: dict[str, Any] = field(default_factory=dict)
    metrics_before: dict[str, Any] = field(default_factory=dict)
    metrics_after: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.adapter_id:
            raise ValueError("AdaptationPacket.adapter_id cannot be empty")
        if not self.update_type:
            raise ValueError("AdaptationPacket.update_type cannot be empty")
        if self.confidence_gate is not None and not 0.0 <= self.confidence_gate <= 1.0:
            raise ValueError("AdaptationPacket.confidence_gate must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "update_type": self.update_type,
            "input_window_ids": list(self.input_window_ids),
            "labels": list(self.labels),
            "confidence_gate": self.confidence_gate,
            "parameters_changed": self.parameters_changed,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "metadata": self.metadata,
        }
