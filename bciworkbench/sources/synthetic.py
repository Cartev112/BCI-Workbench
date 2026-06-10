from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bciworkbench.ontology.packets import ChannelSchema, Event, SignalPacket
from bciworkbench.sim.profiles import SessionProfile, SubjectProfile


@dataclass(frozen=True)
class SyntheticMotorImageryConfig:
    duration_s: float = 120.0
    sampling_rate: float = 250.0
    n_channels: int = 16
    n_trials: int = 80
    trial_duration_s: float = 2.5
    inter_trial_s: float = 0.5
    snr_db: float = -4.0
    line_noise_hz: float = 60.0
    seed: int = 0
    subject: SubjectProfile = field(default_factory=SubjectProfile)
    session: SessionProfile = field(default_factory=SessionProfile)


class SyntheticMotorImagerySource:
    """Deterministic motor imagery-like EEG source.

    The generator is intentionally modest: it creates EEG-shaped rhythms plus
    class-specific mu/beta modulation and trial events. It is a milestone
    source for exercising ontology, runtime, metrics, and reports, not a
    validated physiological model.
    """

    def __init__(self, config: SyntheticMotorImageryConfig) -> None:
        self.config = config

    @classmethod
    def from_params(cls, params: dict, seed: int) -> "SyntheticMotorImagerySource":
        merged = dict(params)
        subject = SubjectProfile.from_dict(merged.pop("subject", None))
        session_values = merged.pop("session", None) or {}
        if "drift" in merged:
            session_values = {**session_values, "amplitude_drift": merged.pop("drift")}
        session = SessionProfile.from_dict(session_values)
        merged.setdefault("seed", seed)
        merged["subject"] = subject
        merged["session"] = session
        return cls(SyntheticMotorImageryConfig(**merged))

    def read(self) -> SignalPacket:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        n_samples = int(cfg.duration_s * cfg.sampling_rate)
        timestamps = np.arange(n_samples, dtype=float) / cfg.sampling_rate
        channel_names = tuple(f"EEG{index + 1:02d}" for index in range(cfg.n_channels))
        bad_channels = self._dropout_channels(rng, channel_names)
        channel_schema = ChannelSchema(
            names=channel_names,
            types=tuple("eeg" for _ in channel_names),
            units=tuple("V" for _ in channel_names),
            sampling_rate=cfg.sampling_rate,
            reference="synthetic_average",
            montage="synthetic_10_20_subset",
            bad_channels=bad_channels,
            metadata={
                "simulation_level": "2_paradigm_with_subject_session_stressors",
                "subject": cfg.subject.to_dict(),
                "session": cfg.session.to_dict(),
            },
        )

        data = self._background_eeg(rng, timestamps, cfg.n_channels)
        events: list[Event] = []

        left_pattern, right_pattern = self._class_patterns(cfg.n_channels, cfg.session.electrode_shift_mm)
        trial_spacing = cfg.trial_duration_s + cfg.inter_trial_s
        max_trials = min(cfg.n_trials, int((cfg.duration_s - cfg.trial_duration_s) // trial_spacing))
        signal_power = float(np.mean(data**2))
        snr_linear = 10 ** (cfg.snr_db / 10.0)
        effect_scale = (
            np.sqrt(max(signal_power * snr_linear, 1e-16))
            * cfg.subject.motor_imagery_vividness
            * cfg.subject.attention
        )

        for trial_index in range(max_trials):
            label = "left" if trial_index % 2 == 0 else "right"
            true_start_s = 1.0 + trial_index * trial_spacing
            marker_jitter_s = rng.normal(0.0, cfg.session.marker_jitter_ms / 1000.0) if cfg.session.marker_jitter_ms else 0.0
            start_s = max(0.0, true_start_s + marker_jitter_s)
            end_s = start_s + cfg.trial_duration_s
            sample_start = int(start_s * cfg.sampling_rate)
            sample_end = min(int(end_s * cfg.sampling_rate), n_samples)
            if sample_end <= sample_start:
                continue

            pattern = left_pattern if label == "left" else right_pattern
            trial_t = timestamps[sample_start:sample_end] - start_s
            envelope = np.sin(np.pi * trial_t / cfg.trial_duration_s) ** 2
            fatigue = max(0.0, 1.0 - cfg.subject.fatigue_rate * trial_index)
            mu = np.sin(2 * np.pi * cfg.subject.alpha_peak_hz * timestamps[sample_start:sample_end])
            beta = 0.5 * np.sin(2 * np.pi * cfg.subject.beta_peak_hz * timestamps[sample_start:sample_end])
            modulation = effect_scale * fatigue * envelope * (mu + beta)
            data[:, sample_start:sample_end] += pattern[:, None] * modulation[None, :]

            event_id = f"trial-{trial_index:04d}"
            events.append(
                Event(
                    event_id=event_id,
                    event_type="trial.start",
                    name="motor_imagery_trial",
                    onset=start_s,
                    duration=cfg.trial_duration_s,
                    sample_index=sample_start,
                    target=label,
                    source="synthetic_motor_imagery",
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
                    event_id=f"{event_id}-cue",
                    event_type="cue.onset",
                    name=f"cue_{label}",
                    onset=start_s,
                    duration=0.0,
                    sample_index=sample_start,
                    target=label,
                    source="synthetic_motor_imagery",
                    metadata={
                        "trial_index": trial_index,
                        "label": label,
                        "true_onset": true_start_s,
                        "marker_jitter_ms": marker_jitter_s * 1000.0,
                    },
                )
            )

        artifact_events = self._apply_artifacts(rng, data, timestamps)
        events.extend(artifact_events)

        dropout_indices = [channel_names.index(name) for name in bad_channels]
        for index in dropout_indices:
            data[index, :] *= 0.05

        drift_curve = 1.0 + cfg.session.amplitude_drift * np.linspace(-0.5, 0.5, n_samples)
        data *= drift_curve[None, :]

        return SignalPacket(
            data=data,
            timestamps=timestamps,
            channel_schema=channel_schema,
            modality="EEG",
            events=events,
            source_id="synthetic_motor_imagery",
            metadata={
                "duration_s": cfg.duration_s,
                "n_trials": max_trials,
                "snr_db": cfg.snr_db,
                "simulation_level": "2_paradigm_with_subject_session_stressors",
                "subject": cfg.subject.to_dict(),
                "session": cfg.session.to_dict(),
                "artifact_event_count": len(artifact_events),
                "bad_channels": list(bad_channels),
            },
        )

    def _background_eeg(self, rng: np.random.Generator, timestamps: np.ndarray, n_channels: int) -> np.ndarray:
        cfg = self.config
        n_samples = timestamps.size
        data = rng.normal(0.0, 2.0e-6, size=(n_channels, n_samples))
        spatial = rng.normal(0.0, 1.0, size=(n_channels, 3))
        sources = np.vstack(
            [
                np.sin(2 * np.pi * 6.0 * timestamps + rng.uniform(0, 2 * np.pi)),
                np.sin(2 * np.pi * cfg.subject.alpha_peak_hz * timestamps + rng.uniform(0, 2 * np.pi)),
                np.sin(2 * np.pi * cfg.subject.beta_peak_hz * timestamps + rng.uniform(0, 2 * np.pi)),
            ]
        )
        data += 5.0e-6 * spatial @ sources
        data += 0.8e-6 * np.sin(2 * np.pi * cfg.line_noise_hz * timestamps)[None, :]
        return data

    @staticmethod
    def _class_patterns(n_channels: int, electrode_shift_mm: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        axis = np.linspace(-1.0, 1.0, n_channels)
        shift = float(np.clip(electrode_shift_mm / 30.0, -0.5, 0.5))
        left = np.exp(-((axis + 0.45 - shift) ** 2) / 0.18)
        right = np.exp(-((axis - 0.45 - shift) ** 2) / 0.18)
        left = left / np.linalg.norm(left)
        right = right / np.linalg.norm(right)
        return left, right

    def _dropout_channels(self, rng: np.random.Generator, channel_names: tuple[str, ...]) -> tuple[str, ...]:
        probability = self.config.session.channel_dropout_probability
        if probability <= 0:
            return ()
        mask = rng.random(len(channel_names)) < probability
        if np.all(mask):
            mask[rng.integers(0, len(channel_names))] = False
        return tuple(name for name, dropped in zip(channel_names, mask, strict=True) if dropped)

    def _apply_artifacts(self, rng: np.random.Generator, data: np.ndarray, timestamps: np.ndarray) -> list[Event]:
        events: list[Event] = []
        events.extend(self._apply_blinks(rng, data, timestamps))
        events.extend(self._apply_muscle_noise(rng, data, timestamps))
        return events

    def _apply_blinks(self, rng: np.random.Generator, data: np.ndarray, timestamps: np.ndarray) -> list[Event]:
        cfg = self.config
        if cfg.session.blink_rate_per_min <= 0:
            return []
        expected = cfg.session.blink_rate_per_min * cfg.duration_s / 60.0
        n_blinks = int(rng.poisson(expected))
        frontal = np.linspace(1.0, 0.15, data.shape[0])
        events: list[Event] = []
        width_s = 0.18
        for index in range(n_blinks):
            onset = float(rng.uniform(0.5, max(0.6, cfg.duration_s - 0.5)))
            shape = np.exp(-0.5 * ((timestamps - onset) / width_s) ** 2)
            amplitude = cfg.session.blink_amplitude_uv * 1e-6 * rng.uniform(0.7, 1.3)
            data += frontal[:, None] * amplitude * shape[None, :]
            sample_index = int(onset * cfg.sampling_rate)
            events.append(
                Event(
                    event_id=f"artifact-blink-{index:04d}",
                    event_type="artifact.blink",
                    name="synthetic_blink",
                    onset=onset,
                    duration=width_s * 4.0,
                    sample_index=sample_index,
                    source="synthetic_motor_imagery",
                    metadata={"amplitude_uv": amplitude * 1e6},
                )
            )
        return events

    def _apply_muscle_noise(self, rng: np.random.Generator, data: np.ndarray, timestamps: np.ndarray) -> list[Event]:
        cfg = self.config
        if cfg.session.muscle_noise <= 0:
            return []
        n_bursts = max(1, int(round(cfg.session.muscle_noise * cfg.duration_s / 20.0)))
        events: list[Event] = []
        for index in range(n_bursts):
            onset = float(rng.uniform(0.5, max(0.6, cfg.duration_s - 0.5)))
            duration = float(rng.uniform(0.2, 0.8))
            start = int(onset * cfg.sampling_rate)
            end = min(data.shape[1], start + int(duration * cfg.sampling_rate))
            if end <= start:
                continue
            burst = rng.normal(0.0, cfg.session.muscle_noise * 4e-6, size=(data.shape[0], end - start))
            high_freq = np.sin(2 * np.pi * rng.uniform(35.0, 80.0) * timestamps[start:end])
            data[:, start:end] += burst * high_freq[None, :]
            events.append(
                Event(
                    event_id=f"artifact-muscle-{index:04d}",
                    event_type="artifact.emg",
                    name="synthetic_muscle_burst",
                    onset=onset,
                    duration=duration,
                    sample_index=start,
                    source="synthetic_motor_imagery",
                    metadata={"severity": cfg.session.muscle_noise},
                )
            )
        return events
