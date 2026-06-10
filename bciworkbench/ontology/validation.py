from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from bciworkbench.ontology.timing import SUPPORTED_CLOCK_DOMAINS


SUPPORTED_CHANNEL_TYPES = {
    "eeg",
    "ecog",
    "ieeg",
    "emg",
    "ecg",
    "eog",
    "fnirs",
    "spike",
    "marker",
    "aux",
}

SUPPORTED_MODALITIES = {"EEG", "ECoG", "iEEG", "EMG", "ECG", "EOG", "fNIRS", "spikes", "aux"}


def require_clock_domain(clock_domain: str) -> None:
    if clock_domain not in SUPPORTED_CLOCK_DOMAINS:
        raise ValueError(f"unsupported clock domain: {clock_domain}")


def require_channel_types(types: Sequence[str]) -> None:
    unsupported = sorted(set(types) - SUPPORTED_CHANNEL_TYPES)
    if unsupported:
        raise ValueError(f"unsupported channel type(s): {', '.join(unsupported)}")


def require_modality(modality: str) -> None:
    if modality not in SUPPORTED_MODALITIES:
        raise ValueError(f"unsupported modality: {modality}")


def require_monotonic_timestamps(timestamps: np.ndarray) -> None:
    if timestamps.size > 1 and bool(np.any(np.diff(timestamps) < 0)):
        raise ValueError("timestamps must be monotonic non-decreasing")

