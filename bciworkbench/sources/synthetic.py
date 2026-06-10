from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bciworkbench.ontology.packets import ChannelSchema, Event, SignalPacket


@dataclass(frozen=True)
class SyntheticMotorImageryConfig:
    duration_s: float = 120.0
    sampling_rate: float = 250.0
    n_channels: int = 16
    n_trials: int = 80
    trial_duration_s: float = 2.5
    inter_trial_s: float = 0.5
    snr_db: float = -4.0
    drift: float = 0.05
    line_noise_hz: float = 60.0
    seed: int = 0


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
        merged.setdefault("seed", seed)
        return cls(SyntheticMotorImageryConfig(**merged))

    def read(self) -> SignalPacket:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        n_samples = int(cfg.duration_s * cfg.sampling_rate)
        timestamps = np.arange(n_samples, dtype=float) / cfg.sampling_rate
        channel_names = tuple(f"EEG{index + 1:02d}" for index in range(cfg.n_channels))
        channel_schema = ChannelSchema(
            names=channel_names,
            types=tuple("eeg" for _ in channel_names),
            units=tuple("V" for _ in channel_names),
            sampling_rate=cfg.sampling_rate,
            reference="synthetic_average",
            montage="synthetic_10_20_subset",
            metadata={"simulation_level": "1_spectral_plus_mi_effect"},
        )

        data = self._background_eeg(rng, timestamps, cfg.n_channels)
        events: list[Event] = []

        left_pattern, right_pattern = self._class_patterns(cfg.n_channels)
        trial_spacing = cfg.trial_duration_s + cfg.inter_trial_s
        max_trials = min(cfg.n_trials, int((cfg.duration_s - cfg.trial_duration_s) // trial_spacing))
        signal_power = float(np.mean(data**2))
        snr_linear = 10 ** (cfg.snr_db / 10.0)
        effect_scale = np.sqrt(max(signal_power * snr_linear, 1e-16))

        for trial_index in range(max_trials):
            label = "left" if trial_index % 2 == 0 else "right"
            start_s = 1.0 + trial_index * trial_spacing
            end_s = start_s + cfg.trial_duration_s
            sample_start = int(start_s * cfg.sampling_rate)
            sample_end = min(int(end_s * cfg.sampling_rate), n_samples)
            if sample_end <= sample_start:
                continue

            pattern = left_pattern if label == "left" else right_pattern
            trial_t = timestamps[sample_start:sample_end] - start_s
            envelope = np.sin(np.pi * trial_t / cfg.trial_duration_s) ** 2
            mu = np.sin(2 * np.pi * 10.0 * timestamps[sample_start:sample_end])
            beta = 0.5 * np.sin(2 * np.pi * 20.0 * timestamps[sample_start:sample_end])
            modulation = effect_scale * envelope * (mu + beta)
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
                    metadata={"trial_index": trial_index, "label": label},
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
                    metadata={"trial_index": trial_index, "label": label},
                )
            )

        drift_curve = 1.0 + cfg.drift * np.linspace(-0.5, 0.5, n_samples)
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
                "drift": cfg.drift,
                "simulation_level": "1_spectral_plus_mi_effect",
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
                np.sin(2 * np.pi * 10.0 * timestamps + rng.uniform(0, 2 * np.pi)),
                np.sin(2 * np.pi * 22.0 * timestamps + rng.uniform(0, 2 * np.pi)),
            ]
        )
        data += 5.0e-6 * spatial @ sources
        data += 0.8e-6 * np.sin(2 * np.pi * cfg.line_noise_hz * timestamps)[None, :]
        return data

    @staticmethod
    def _class_patterns(n_channels: int) -> tuple[np.ndarray, np.ndarray]:
        axis = np.linspace(-1.0, 1.0, n_channels)
        left = np.exp(-((axis + 0.45) ** 2) / 0.18)
        right = np.exp(-((axis - 0.45) ** 2) / 0.18)
        left = left / np.linalg.norm(left)
        right = right / np.linalg.norm(right)
        return left, right

