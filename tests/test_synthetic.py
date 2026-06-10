from __future__ import annotations

import numpy as np

from bciworkbench.sources.synthetic import SyntheticMotorImageryConfig, SyntheticMotorImagerySource


def test_synthetic_source_is_deterministic() -> None:
    cfg = SyntheticMotorImageryConfig(duration_s=20, n_trials=4, seed=10)
    first = SyntheticMotorImagerySource(cfg).read()
    second = SyntheticMotorImagerySource(cfg).read()
    assert np.allclose(first.data, second.data)
    assert len(first.events) == len(second.events)
    assert first.channel_schema.n_channels == cfg.n_channels


def test_synthetic_source_emits_trial_targets() -> None:
    packet = SyntheticMotorImagerySource(SyntheticMotorImageryConfig(duration_s=20, n_trials=4, seed=2)).read()
    trials = [event for event in packet.events if event.event_type == "trial.start"]
    assert {event.target for event in trials} == {"left", "right"}

