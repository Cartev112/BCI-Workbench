from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bciworkbench.ontology.packets import ChannelSchema, Event, SignalPacket
from bciworkbench.sim.profiles import SessionProfile, SubjectProfile
from bciworkbench.sources.synthetic import SyntheticMotorImagerySource


@dataclass(frozen=True)
class SyntheticP300Config:
    duration_s: float = 120.0
    sampling_rate: float = 250.0
    n_channels: int = 16
    n_trials: int = 120
    trial_duration_s: float = 0.8
    inter_trial_s: float = 0.2
    target_probability: float = 0.25
    snr_db: float = -6.0
    line_noise_hz: float = 60.0
    seed: int = 0
    subject: SubjectProfile = field(default_factory=SubjectProfile)
    session: SessionProfile = field(default_factory=SessionProfile)


class SyntheticP300Source:
    """Paradigm-aware synthetic P300 oddball source."""

    def __init__(self, config: SyntheticP300Config) -> None:
        self.config = config

    @classmethod
    def from_params(cls, params: dict, seed: int) -> "SyntheticP300Source":
        merged = dict(params)
        subject = SubjectProfile.from_dict(merged.pop("subject", None))
        session = SessionProfile.from_dict(merged.pop("session", None) or {})
        merged.setdefault("seed", seed)
        merged["subject"] = subject
        merged["session"] = session
        return cls(SyntheticP300Config(**merged))

    def read(self) -> SignalPacket:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        n_samples = int(cfg.duration_s * cfg.sampling_rate)
        timestamps = np.arange(n_samples, dtype=float) / cfg.sampling_rate
        channel_names = tuple(f"EEG{index + 1:02d}" for index in range(cfg.n_channels))
        helper = SyntheticMotorImagerySource.__new__(SyntheticMotorImagerySource)
        helper.config = cfg
        bad_channels = helper._dropout_channels(rng, channel_names)
        data = helper._background_eeg(rng, timestamps, cfg.n_channels)

        channel_schema = ChannelSchema(
            names=channel_names,
            types=tuple("eeg" for _ in channel_names),
            units=tuple("V" for _ in channel_names),
            sampling_rate=cfg.sampling_rate,
            reference="synthetic_average",
            montage="synthetic_10_20_subset",
            bad_channels=bad_channels,
            metadata={
                "simulation_level": "2_p300_paradigm_with_subject_session_stressors",
                "subject": cfg.subject.to_dict(),
                "session": cfg.session.to_dict(),
            },
        )

        events: list[Event] = []
        posterior_pattern = _posterior_pattern(cfg.n_channels)
        trial_spacing = cfg.trial_duration_s + cfg.inter_trial_s
        max_trials = min(cfg.n_trials, int((cfg.duration_s - cfg.trial_duration_s) // trial_spacing))
        signal_power = float(np.mean(data**2))
        snr_linear = 10 ** (cfg.snr_db / 10.0)
        erp_scale = np.sqrt(max(signal_power * snr_linear, 1e-16))

        for trial_index in range(max_trials):
            label = "target" if rng.random() < cfg.target_probability else "non_target"
            true_start_s = 1.0 + trial_index * trial_spacing
            marker_jitter_s = rng.normal(0.0, cfg.session.marker_jitter_ms / 1000.0) if cfg.session.marker_jitter_ms else 0.0
            start_s = max(0.0, true_start_s + marker_jitter_s)
            sample_start = int(start_s * cfg.sampling_rate)
            sample_end = min(int((start_s + cfg.trial_duration_s) * cfg.sampling_rate), n_samples)
            if sample_end <= sample_start:
                continue
            fatigue = max(0.0, 1.0 - cfg.subject.fatigue_rate * trial_index)
            if label == "target":
                latency_s = max(
                    0.12,
                    (cfg.subject.p300_latency_ms + rng.normal(0.0, cfg.subject.p300_latency_jitter_ms)) / 1000.0,
                )
                center = start_s + latency_s
                width_s = 0.075
                erp = np.exp(-0.5 * ((timestamps[sample_start:sample_end] - center) / width_s) ** 2)
                amplitude = cfg.subject.p300_amplitude_uv * 1e-6 * cfg.subject.attention * fatigue
                data[:, sample_start:sample_end] += posterior_pattern[:, None] * amplitude * erp_scale / 1e-6 * erp[None, :]
            event_id = f"p300-trial-{trial_index:04d}"
            events.append(
                Event(
                    event_id=event_id,
                    event_type="trial.start",
                    name="p300_trial",
                    onset=start_s,
                    duration=cfg.trial_duration_s,
                    sample_index=sample_start,
                    target=label,
                    source="synthetic_p300",
                    metadata={
                        "trial_index": trial_index,
                        "label": label,
                        "true_onset": true_start_s,
                        "marker_jitter_ms": marker_jitter_s * 1000.0,
                        "fatigue_multiplier": fatigue,
                    },
                )
            )
            events.append(
                Event(
                    event_id=f"{event_id}-stimulus",
                    event_type="stimulus.onset",
                    name=f"p300_{label}",
                    onset=start_s,
                    duration=0.0,
                    sample_index=sample_start,
                    target=label,
                    source="synthetic_p300",
                    metadata={"trial_index": trial_index, "label": label},
                )
            )

        artifact_events = helper._apply_artifacts(rng, data, timestamps, source_id="synthetic_p300")
        artifact_events.extend(helper._dropout_events(bad_channels, cfg.duration_s, source_id="synthetic_p300"))
        events.extend(artifact_events)
        for channel in bad_channels:
            data[channel_names.index(channel), :] *= 0.05
        data *= (1.0 + cfg.session.amplitude_drift * np.linspace(-0.5, 0.5, n_samples))[None, :]

        return SignalPacket(
            data=data,
            timestamps=timestamps,
            channel_schema=channel_schema,
            modality="EEG",
            events=events,
            source_id="synthetic_p300",
            metadata={
                "duration_s": cfg.duration_s,
                "n_trials": max_trials,
                "target_probability": cfg.target_probability,
                "snr_db": cfg.snr_db,
                "simulation_level": "2_p300_paradigm_with_subject_session_stressors",
                "subject": cfg.subject.to_dict(),
                "session": cfg.session.to_dict(),
                "artifact_event_count": len(artifact_events),
                "bad_channels": list(bad_channels),
                "feedback_delay_ms": cfg.session.feedback_delay_ms,
            },
        )


def _posterior_pattern(n_channels: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, n_channels)
    pattern = np.exp(-((axis - 0.15) ** 2) / 0.35)
    return pattern / np.linalg.norm(pattern)
