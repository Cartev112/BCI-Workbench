from __future__ import annotations

import numpy as np

from bciworkbench.sim.profiles import SessionProfile, SubjectProfile
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


def test_synthetic_source_logs_subject_session_and_artifacts() -> None:
    cfg = SyntheticMotorImageryConfig(
        duration_s=30,
        n_trials=4,
        seed=3,
        subject=SubjectProfile(alpha_peak_hz=11.0, fatigue_rate=0.01),
        session=SessionProfile(
            blink_rate_per_min=12,
            muscle_noise=0.2,
            channel_dropout_probability=0.2,
            marker_jitter_ms=5.0,
        ),
    )
    packet = SyntheticMotorImagerySource(cfg).read()
    artifact_events = [event for event in packet.events if event.event_type.startswith("artifact.")]
    trial_events = [event for event in packet.events if event.event_type == "trial.start"]
    assert packet.metadata["simulation_level"] == "2_paradigm_with_subject_session_stressors"
    assert packet.metadata["subject"]["alpha_peak_hz"] == 11.0
    assert packet.metadata["session"]["marker_jitter_ms"] == 5.0
    assert packet.metadata["artifact_event_count"] == len(artifact_events)
    assert trial_events[0].metadata["true_onset"] != trial_events[0].onset
