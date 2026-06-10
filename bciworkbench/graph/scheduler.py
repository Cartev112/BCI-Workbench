from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReplayScheduleConfig:
    mode: str = "fastest"
    speed: float = 1.0
    chunk_duration_s: float = 0.25
    step_duration_s: float = 0.0
    processing_time_ms: float = 0.0
    queue_capacity: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"fastest", "real_time", "scaled", "stepped"}:
            raise ValueError("replay mode must be one of: fastest, real_time, scaled, stepped")
        if self.speed <= 0:
            raise ValueError("replay speed must be positive")
        if self.chunk_duration_s <= 0:
            raise ValueError("chunk_duration_s must be positive")
        if self.step_duration_s < 0:
            raise ValueError("step_duration_s must be non-negative")
        if self.processing_time_ms < 0:
            raise ValueError("processing_time_ms must be non-negative")
        if self.queue_capacity is not None and self.queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive when provided")


@dataclass(frozen=True)
class ReplayTraceRow:
    packet_index: int
    sample_start: int
    sample_end: int
    signal_time_start_s: float
    signal_time_end_s: float
    scheduled_arrival_s: float
    arrival_time_s: float
    arrival_delay_ms: float
    backlog_ms: float
    queue_depth: int
    dropped: bool
    sleep_s: float
    speed_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReplayScheduler:
    """Deterministic replay packet scheduler.

    The scheduler separates original signal time from simulated packet arrival
    time. It does not require async runtime support; sources can use the trace
    as replay telemetry while still returning a full SignalPacket to the
    current linear graph.
    """

    def __init__(self, config: ReplayScheduleConfig | None = None) -> None:
        self.config = config or ReplayScheduleConfig()

    def schedule(self, timestamps: np.ndarray, sampling_rate: float) -> list[ReplayTraceRow]:
        if timestamps.ndim != 1 or timestamps.size == 0:
            raise ValueError("timestamps must be a non-empty one-dimensional array")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        cfg = self.config
        chunk_samples = max(1, int(round(cfg.chunk_duration_s * sampling_rate)))
        first_time = float(timestamps[0])
        previous_arrival = 0.0
        previous_processing_done = 0.0
        rows: list[ReplayTraceRow] = []

        for packet_index, sample_start in enumerate(range(0, timestamps.size, chunk_samples)):
            sample_end = min(sample_start + chunk_samples, timestamps.size)
            signal_start = float(timestamps[sample_start])
            signal_end = float(timestamps[sample_end - 1])
            scheduled = self._scheduled_arrival(packet_index, signal_start - first_time)
            sleep_s = max(0.0, scheduled - previous_arrival)
            arrival = scheduled
            processing_start = max(arrival, previous_processing_done)
            backlog_s = max(0.0, processing_start - arrival)
            interval_s = _packet_interval_s(rows, scheduled, cfg.chunk_duration_s)
            queue_depth = int(np.ceil(backlog_s / interval_s)) if backlog_s > 0 else 0
            dropped = cfg.queue_capacity is not None and queue_depth > cfg.queue_capacity
            previous_processing_done = processing_start + cfg.processing_time_ms / 1000.0
            previous_arrival = arrival
            rows.append(
                ReplayTraceRow(
                    packet_index=packet_index,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    signal_time_start_s=signal_start,
                    signal_time_end_s=signal_end,
                    scheduled_arrival_s=scheduled,
                    arrival_time_s=arrival,
                    arrival_delay_ms=(arrival - scheduled) * 1000.0,
                    backlog_ms=backlog_s * 1000.0,
                    queue_depth=queue_depth,
                    dropped=dropped,
                    sleep_s=sleep_s,
                    speed_mode=cfg.mode,
                )
            )
        return rows

    def health(self, rows: list[ReplayTraceRow]) -> dict[str, Any]:
        if not rows:
            return {
                "packet_count": 0,
                "dropped_packets": 0,
                "max_backlog_ms": 0.0,
                "mean_arrival_delay_ms": 0.0,
                "max_queue_depth": 0,
                "speed_mode": self.config.mode,
                "speed": self.config.speed,
            }
        return {
            "packet_count": len(rows),
            "dropped_packets": sum(1 for row in rows if row.dropped),
            "max_backlog_ms": max(row.backlog_ms for row in rows),
            "mean_arrival_delay_ms": float(np.mean([row.arrival_delay_ms for row in rows])),
            "max_queue_depth": max(row.queue_depth for row in rows),
            "speed_mode": self.config.mode,
            "speed": self.config.speed,
        }

    def _scheduled_arrival(self, packet_index: int, signal_elapsed_s: float) -> float:
        cfg = self.config
        if cfg.mode == "fastest":
            return 0.0
        if cfg.mode == "real_time":
            return signal_elapsed_s
        if cfg.mode == "scaled":
            return signal_elapsed_s / cfg.speed
        step = cfg.step_duration_s if cfg.step_duration_s > 0 else cfg.chunk_duration_s
        return packet_index * step


def _packet_interval_s(rows: list[ReplayTraceRow], scheduled: float, fallback: float) -> float:
    if not rows:
        return max(fallback, 1e-9)
    return max(scheduled - rows[-1].scheduled_arrival_s, 1e-9)
