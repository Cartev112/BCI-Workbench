from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bciworkbench.ontology.packets import ChannelSchema, Event, SignalPacket
from bciworkbench.sources.base import OptionalDependencyError


@dataclass(frozen=True)
class MNERawSourceConfig:
    path: Path
    preload: bool = True
    event_id_prefix: str = "mne-annotation"


class MNERawSource:
    """Read an MNE Raw-compatible FIF file into a SignalPacket."""

    def __init__(self, config: MNERawSourceConfig) -> None:
        self.config = config
        self.raw = None

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "MNERawSource":
        path = params.get("path")
        if not path:
            raise ValueError("mne_raw source requires source.path")
        return cls(
            MNERawSourceConfig(
                path=Path(str(path)),
                preload=bool(params.get("preload", True)),
                event_id_prefix=str(params.get("event_id_prefix", "mne-annotation")),
            )
        )

    def read(self) -> SignalPacket:
        try:
            import mne
        except Exception as exc:
            raise OptionalDependencyError(
                'mne_raw source requires bciworkbench[mne]. Install with: pip install "bciworkbench[mne]"'
            ) from exc

        raw = mne.io.read_raw_fif(self.config.path, preload=self.config.preload, verbose="ERROR")
        self.raw = raw
        data = raw.get_data()
        timestamps = raw.times.astype(float)
        channel_schema = _channel_schema_from_raw(raw)
        events = _events_from_annotations(raw, self.config.event_id_prefix)
        return SignalPacket(
            data=data,
            timestamps=timestamps,
            channel_schema=channel_schema,
            modality="EEG",
            events=events,
            clock_domain="recording_clock",
            source_id="mne_raw",
            metadata={
                "path": str(self.config.path),
                "n_times": int(raw.n_times),
                "measurement_date": str(raw.info.get("meas_date")),
                "source_format": "mne_raw_fif",
            },
        )


def _channel_schema_from_raw(raw) -> ChannelSchema:
    names = tuple(str(name) for name in raw.ch_names)
    types = tuple(_normalize_channel_type(kind) for kind in raw.get_channel_types())
    units = tuple("V" for _ in names)
    montage = raw.get_montage()
    metadata: dict[str, Any] = {
        "mne_info_keys": sorted(str(key) for key in raw.info.keys()),
        "highpass": float(raw.info.get("highpass", 0.0)),
        "lowpass": float(raw.info.get("lowpass", 0.0)),
    }
    return ChannelSchema(
        names=names,
        types=types,
        units=units,
        sampling_rate=float(raw.info["sfreq"]),
        reference="mne_raw",
        montage=montage.__class__.__name__ if montage is not None else None,
        bad_channels=tuple(str(name) for name in raw.info.get("bads", [])),
        metadata=metadata,
    )


def _events_from_annotations(raw, prefix: str) -> list[Event]:
    events: list[Event] = []
    for index, annotation in enumerate(raw.annotations):
        onset = float(annotation["onset"])
        sample_index = int(np.searchsorted(raw.times, onset))
        events.append(
            Event(
                event_id=f"{prefix}-{index:04d}",
                event_type=_annotation_to_event_type(str(annotation["description"])),
                name=str(annotation["description"]),
                onset=onset,
                duration=float(annotation["duration"]),
                clock_domain="recording_clock",
                sample_index=sample_index,
                source="mne.annotations",
                target=str(annotation["description"]),
                metadata={"orig_time": str(raw.annotations.orig_time)},
            )
        )
    return events


def _annotation_to_event_type(description: str) -> str:
    lowered = description.lower()
    if "cue" in lowered:
        return "cue.onset"
    if "trial" in lowered:
        return "trial.start"
    if "target" in lowered:
        return "target.presented"
    return "stimulus.onset"


def _normalize_channel_type(kind: str) -> str:
    mapping = {
        "eeg": "eeg",
        "ecog": "ecog",
        "seeg": "ieeg",
        "dbs": "ieeg",
        "emg": "emg",
        "ecg": "ecg",
        "eog": "eog",
        "fnirs_cw_amplitude": "fnirs",
        "fnirs_fd_ac_amplitude": "fnirs",
        "fnirs_fd_phase": "fnirs",
        "fnirs_od": "fnirs",
        "hbo": "fnirs",
        "hbr": "fnirs",
    }
    return mapping.get(kind, "aux")

