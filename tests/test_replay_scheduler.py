from __future__ import annotations

import numpy as np

from bciworkbench.graph.scheduler import ReplayScheduleConfig, ReplayScheduler


def test_replay_scheduler_fastest_keeps_all_arrivals_at_zero() -> None:
    timestamps = np.arange(100, dtype=float) / 100.0
    rows = ReplayScheduler(ReplayScheduleConfig(mode="fastest", chunk_duration_s=0.25)).schedule(
        timestamps,
        sampling_rate=100.0,
    )
    assert len(rows) == 4
    assert {row.arrival_time_s for row in rows} == {0.0}
    assert rows[0].sample_start == 0
    assert rows[-1].sample_end == 100


def test_replay_scheduler_scaled_mode_speeds_original_timing() -> None:
    timestamps = np.arange(100, dtype=float) / 100.0
    rows = ReplayScheduler(ReplayScheduleConfig(mode="scaled", speed=2.0, chunk_duration_s=0.25)).schedule(
        timestamps,
        sampling_rate=100.0,
    )
    assert rows[1].scheduled_arrival_s == 0.125
    assert rows[2].scheduled_arrival_s == 0.25


def test_replay_scheduler_reports_backlog_and_queue_depth() -> None:
    timestamps = np.arange(100, dtype=float) / 100.0
    scheduler = ReplayScheduler(
        ReplayScheduleConfig(
            mode="real_time",
            chunk_duration_s=0.25,
            processing_time_ms=400.0,
            queue_capacity=1,
        )
    )
    rows = scheduler.schedule(timestamps, sampling_rate=100.0)
    health = scheduler.health(rows)
    assert health["max_backlog_ms"] > 0
    assert health["max_queue_depth"] > 0
    assert health["dropped_packets"] >= 1
