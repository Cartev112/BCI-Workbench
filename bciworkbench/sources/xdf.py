from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bciworkbench.graph.scheduler import ReplayScheduleConfig, ReplayScheduler
from bciworkbench.ontology.packets import ChannelSchema, Event, SignalPacket
from bciworkbench.sources.base import OptionalDependencyError


@dataclass(frozen=True)
class XDFReplaySourceConfig:
    path: Path
    signal_stream: str | None = None
    marker_stream: str | None = None
    speed_mode: str = "fastest"
    speed: float = 1.0
    chunk_duration_s: float = 0.25
    step_duration_s: float = 0.0
    processing_time_ms: float = 0.0
    queue_capacity: int | None = None


class XDFReplaySource:
    """Replay an XDF recording through deterministic packet timing telemetry."""

    def __init__(self, config: XDFReplaySourceConfig) -> None:
        self.config = config

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "XDFReplaySource":
        path = params.get("path")
        if not path:
            raise ValueError("xdf_replay source requires source.path")
        queue_capacity = params.get("queue_capacity")
        return cls(
            XDFReplaySourceConfig(
                path=Path(str(path)),
                signal_stream=_optional_str(params.get("signal_stream")),
                marker_stream=_optional_str(params.get("marker_stream")),
                speed_mode=str(params.get("speed_mode", "fastest")),
                speed=float(params.get("speed", 1.0)),
                chunk_duration_s=float(params.get("chunk_duration_s", 0.25)),
                step_duration_s=float(params.get("step_duration_s", 0.0)),
                processing_time_ms=float(params.get("processing_time_ms", 0.0)),
                queue_capacity=int(queue_capacity) if queue_capacity is not None else None,
            )
        )

    def read(self) -> SignalPacket:
        streams = _load_streams(self.config.path)
        signal_stream = _select_signal_stream(streams, self.config.signal_stream)
        marker_stream = _select_marker_stream(streams, self.config.marker_stream)
        data = _stream_data(signal_stream)
        raw_timestamps = _stream_timestamps(signal_stream, data.shape[1])
        clock_offset_s = float(raw_timestamps[0])
        timestamps = raw_timestamps - clock_offset_s
        sampling_rate = _sampling_rate(signal_stream, raw_timestamps)
        channel_names = _channel_names(signal_stream, data.shape[0])
        channel_schema = ChannelSchema(
            names=channel_names,
            types=tuple(_channel_types(signal_stream, len(channel_names))),
            units=tuple(_channel_units(signal_stream, len(channel_names))),
            sampling_rate=sampling_rate,
            reference="xdf",
            montage=None,
            metadata={
                "stream_name": _stream_name(signal_stream),
                "stream_type": _stream_type(signal_stream),
                "nominal_srate": _stream_info_value(signal_stream, "nominal_srate"),
                "source_id": _stream_info_value(signal_stream, "source_id"),
            },
        )
        events = _events_from_marker_stream(marker_stream, timestamps, clock_offset_s, self.config.path)
        scheduler = ReplayScheduler(
            ReplayScheduleConfig(
                mode=self.config.speed_mode,
                speed=self.config.speed,
                chunk_duration_s=self.config.chunk_duration_s,
                step_duration_s=self.config.step_duration_s,
                processing_time_ms=self.config.processing_time_ms,
                queue_capacity=self.config.queue_capacity,
            )
        )
        trace = scheduler.schedule(timestamps, sampling_rate)
        stream_health = scheduler.health(trace)
        replay = {
            "speed_mode": self.config.speed_mode,
            "speed": self.config.speed,
            "chunk_duration_s": self.config.chunk_duration_s,
            "step_duration_s": self.config.step_duration_s,
            "processing_time_ms": self.config.processing_time_ms,
            "queue_capacity": self.config.queue_capacity,
            "trace_rows": len(trace),
        }
        return SignalPacket(
            data=data,
            timestamps=timestamps,
            channel_schema=channel_schema,
            modality="EEG",
            events=events,
            clock_domain="recording_clock",
            source_id="xdf_replay",
            metadata={
                "path": str(self.config.path),
                "source_format": "xdf_replay",
                "clock_offset_s": clock_offset_s,
                "signal_stream": _stream_summary(signal_stream),
                "marker_stream": _stream_summary(marker_stream) if marker_stream else None,
                "replay": replay,
                "stream_health": stream_health,
                "latency_trace": [row.to_dict() for row in trace],
            },
        )


def _load_streams(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing XDF replay file: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        streams = payload.get("streams")
        if not isinstance(streams, list) or not streams:
            raise ValueError("XDF JSON fixture must contain a non-empty streams list")
        return streams
    try:
        import pyxdf
    except Exception as exc:
        raise OptionalDependencyError(
            'xdf_replay source requires pyxdf for .xdf files. Install with: pip install "pyxdf"'
        ) from exc
    streams, _header = pyxdf.load_xdf(str(path))
    return list(streams)


def _select_signal_stream(streams: list[dict[str, Any]], requested: str | None) -> dict[str, Any]:
    if requested:
        matches = [stream for stream in streams if _stream_name(stream) == requested]
        if not matches:
            raise ValueError(f"signal stream not found: {requested}")
        return matches[0]
    eeg = [stream for stream in streams if _stream_type(stream).lower() == "eeg"]
    if eeg:
        return eeg[0]
    numeric = [stream for stream in streams if _looks_numeric_series(stream.get("time_series"))]
    if numeric:
        return numeric[0]
    raise ValueError("no numeric signal stream found in XDF replay file")


def _select_marker_stream(streams: list[dict[str, Any]], requested: str | None) -> dict[str, Any] | None:
    if requested:
        matches = [stream for stream in streams if _stream_name(stream) == requested]
        if not matches:
            raise ValueError(f"marker stream not found: {requested}")
        return matches[0]
    marker = [stream for stream in streams if _stream_type(stream).lower() in {"markers", "marker"}]
    return marker[0] if marker else None


def _stream_data(stream: dict[str, Any]) -> np.ndarray:
    series = np.asarray(stream.get("time_series"), dtype=float)
    if series.ndim == 1:
        series = series[:, None]
    if series.ndim != 2:
        raise ValueError("signal stream time_series must be 1D or 2D")
    return series.T.astype(float, copy=False)


def _stream_timestamps(stream: dict[str, Any], n_samples: int) -> np.ndarray:
    raw = stream.get("time_stamps")
    if raw is None:
        sampling_rate = float(_stream_info_value(stream, "nominal_srate") or 0.0)
        if sampling_rate <= 0:
            raise ValueError("signal stream requires time_stamps or a positive nominal_srate")
        return np.arange(n_samples, dtype=float) / sampling_rate
    timestamps = np.asarray(raw, dtype=float)
    if timestamps.ndim != 1 or timestamps.size != n_samples:
        raise ValueError("signal stream time_stamps must be one timestamp per sample")
    return timestamps


def _sampling_rate(stream: dict[str, Any], timestamps: np.ndarray) -> float:
    nominal = _stream_info_value(stream, "nominal_srate")
    if nominal not in {None, ""} and float(nominal) > 0:
        return float(nominal)
    if timestamps.size < 2:
        raise ValueError("cannot infer sampling rate from fewer than two timestamps")
    return float(1.0 / np.median(np.diff(timestamps)))


def _events_from_marker_stream(
    marker_stream: dict[str, Any] | None,
    signal_timestamps: np.ndarray,
    clock_offset_s: float,
    path: Path,
) -> list[Event]:
    if marker_stream is None:
        return []
    stamps = np.asarray(marker_stream.get("time_stamps"), dtype=float)
    labels = marker_stream.get("time_series") or []
    events: list[Event] = []
    for index, (stamp, label_raw) in enumerate(zip(stamps, labels, strict=False)):
        label = _marker_label(label_raw)
        onset = float(stamp) - clock_offset_s
        events.append(
            Event(
                event_id=f"xdf-marker-{index:04d}",
                event_type=_marker_event_type(label),
                name=label,
                onset=max(0.0, onset),
                duration=0.0,
                clock_domain="recording_clock",
                sample_index=int(np.searchsorted(signal_timestamps, onset)),
                source="xdf.marker_stream",
                target=label,
                metadata={
                    "path": str(path),
                    "stream_name": _stream_name(marker_stream),
                    "original_marker_time_s": float(stamp),
                },
            )
        )
    return events


def _marker_event_type(label: str) -> str:
    lowered = label.lower()
    if "trial" in lowered or lowered in {"left", "right", "target", "non_target"}:
        return "trial.start"
    if "cue" in lowered:
        return "cue.onset"
    return "stimulus.onset"


def _marker_label(label: Any) -> str:
    if isinstance(label, list | tuple):
        if not label:
            return ""
        return str(label[0])
    return str(label)


def _channel_names(stream: dict[str, Any], n_channels: int) -> tuple[str, ...]:
    names = stream.get("channel_names")
    if names and len(names) == n_channels:
        return tuple(str(name) for name in names)
    channels = _stream_info_value(stream, "channels")
    if isinstance(channels, list) and len(channels) == n_channels:
        return tuple(str(channel.get("label", f"ch{index + 1}")) for index, channel in enumerate(channels))
    return tuple(f"EEG{index + 1:02d}" for index in range(n_channels))


def _channel_types(stream: dict[str, Any], n_channels: int) -> list[str]:
    values = stream.get("channel_types")
    if values and len(values) == n_channels:
        return [str(value).lower() for value in values]
    return ["eeg"] * n_channels


def _channel_units(stream: dict[str, Any], n_channels: int) -> list[str]:
    values = stream.get("channel_units")
    if values and len(values) == n_channels:
        return [str(value) for value in values]
    return ["V"] * n_channels


def _stream_summary(stream: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _stream_name(stream),
        "type": _stream_type(stream),
        "nominal_srate": _stream_info_value(stream, "nominal_srate"),
        "sample_count": len(stream.get("time_stamps") or stream.get("time_series") or []),
    }


def _stream_name(stream: dict[str, Any]) -> str:
    return str(_stream_info_value(stream, "name") or stream.get("name") or "")


def _stream_type(stream: dict[str, Any]) -> str:
    return str(_stream_info_value(stream, "type") or stream.get("type") or "")


def _stream_info_value(stream: dict[str, Any], key: str) -> Any:
    if key in stream:
        return stream[key]
    info = stream.get("info") or {}
    value = info.get(key)
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _looks_numeric_series(series: Any) -> bool:
    try:
        np.asarray(series, dtype=float)
    except (TypeError, ValueError):
        return False
    return True


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
