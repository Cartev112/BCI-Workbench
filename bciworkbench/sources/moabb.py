from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bciworkbench.ontology.packets import ChannelSchema, Event, SignalPacket
from bciworkbench.sources.base import OptionalDependencyError


@dataclass(frozen=True)
class MOABBSourceConfig:
    dataset: str
    subject: int = 1
    paradigm: str = "motor_imagery"


class MOABBSource:
    """MOABB dataset source.

    The initial implementation supports BNCI2014_001 through MOABB's dataset
    API. Loading may download data, so tests guard network-heavy reads.
    """

    def __init__(self, config: MOABBSourceConfig) -> None:
        self.config = config

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "MOABBSource":
        dataset = params.get("dataset")
        if not dataset:
            raise ValueError("moabb source requires source.dataset")
        return cls(
            MOABBSourceConfig(
                dataset=str(dataset),
                subject=int(params.get("subject", 1)),
                paradigm=str(params.get("paradigm", "motor_imagery")),
            )
        )

    def read(self) -> SignalPacket:
        try:
            from moabb.datasets import BNCI2014_001
        except Exception as exc:
            raise OptionalDependencyError(
                'moabb source requires bciworkbench[moabb]. Install with: pip install "bciworkbench[moabb]"'
            ) from exc
        if self.config.dataset != "BNCI2014_001":
            raise ValueError("only MOABB dataset BNCI2014_001 is supported in this milestone")

        dataset = BNCI2014_001()
        data = dataset.get_data(subjects=[self.config.subject])
        first_session = next(iter(data[self.config.subject].values()))
        first_run = next(iter(first_session.values()))
        raw = first_run
        signal = raw.get_data()
        timestamps = raw.times.astype(float)
        annotations = raw.annotations
        channel_schema = ChannelSchema(
            names=tuple(str(name) for name in raw.ch_names),
            types=tuple("eeg" for _ in raw.ch_names),
            units=tuple("V" for _ in raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]),
            reference="moabb",
            montage="moabb_dataset",
            bad_channels=tuple(str(name) for name in raw.info.get("bads", [])),
            metadata={"dataset": self.config.dataset, "subject": self.config.subject},
        )
        events = [
            Event(
                event_id=f"moabb-annotation-{index:04d}",
                event_type="trial.start",
                name=str(annotation["description"]),
                onset=float(annotation["onset"]),
                duration=float(annotation["duration"]),
                clock_domain="recording_clock",
                sample_index=int(np.searchsorted(raw.times, float(annotation["onset"]))),
                source="moabb.annotations",
                target=str(annotation["description"]),
                metadata={"dataset": self.config.dataset, "subject": self.config.subject},
            )
            for index, annotation in enumerate(annotations)
        ]
        return SignalPacket(
            data=signal,
            timestamps=timestamps,
            channel_schema=channel_schema,
            modality="EEG",
            events=events,
            clock_domain="recording_clock",
            source_id="moabb",
            metadata={
                "dataset": self.config.dataset,
                "subject": self.config.subject,
                "paradigm": self.config.paradigm,
                "source_format": "moabb",
            },
        )

