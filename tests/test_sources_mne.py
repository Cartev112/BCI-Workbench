from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bciworkbench.experiment import Experiment
from bciworkbench.ontology.schemas import parse_experiment_spec
from bciworkbench.sources.mne import MNERawSource


mne = pytest.importorskip("mne")


def _write_mne_raw(path: Path) -> None:
    sfreq = 100.0
    duration_s = 30.0
    n_samples = int(sfreq * duration_s)
    times = np.arange(n_samples) / sfreq
    data = np.vstack(
        [
            1e-6 * np.sin(2 * np.pi * 10 * times),
            1e-6 * np.sin(2 * np.pi * 11 * times),
            1e-6 * np.sin(2 * np.pi * 20 * times),
            1e-6 * np.sin(2 * np.pi * 21 * times),
        ]
    )
    info = mne.create_info(["C3", "C4", "P3", "P4"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    onsets = [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0]
    labels = ["left", "right"] * 4
    raw.set_annotations(
        mne.Annotations(
            onset=onsets,
            duration=[2.0] * len(onsets),
            description=[f"trial_{label}" for label in labels],
        )
    )
    raw.save(path, overwrite=True, verbose="ERROR")


def test_mne_raw_source_converts_channels_events_and_metadata(tmp_path: Path) -> None:
    raw_path = tmp_path / "sample_raw.fif"
    _write_mne_raw(raw_path)

    packet = MNERawSource.from_params({"path": str(raw_path)}).read()

    assert packet.source_id == "mne_raw"
    assert packet.clock_domain == "recording_clock"
    assert packet.channel_schema.names == ("C3", "C4", "P3", "P4")
    assert packet.channel_schema.types == ("eeg", "eeg", "eeg", "eeg")
    assert {event.event_type for event in packet.events} == {"trial.start"}
    assert packet.metadata["source_format"] == "mne_raw_fif"


def test_mne_raw_source_runs_existing_pipeline(tmp_path: Path) -> None:
    raw_path = tmp_path / "pipeline_raw.fif"
    _write_mne_raw(raw_path)
    config = {
        "schema_version": "0.1",
        "name": "mne_raw_pipeline",
        "paradigm": "motor_imagery",
        "mode": "offline",
        "random_seed": 1,
        "output_dir": str(tmp_path / "runs"),
        "source": {"type": "mne_raw", "path": str(raw_path)},
        "pipeline": [
            {"type": "window", "length_s": 1.0, "offset_s": 0.1},
            {"type": "bandpower"},
            {"type": "decoder", "estimator": "nearest_centroid", "calibration_fraction": 0.5},
        ],
        "task": {"type": "motor_imagery_classification", "classes": ["trial_left", "trial_right"]},
    }
    result = Experiment(parse_experiment_spec(config)).run()
    assert result.metrics["source"] == "mne_raw"
    assert result.metrics["n_predictions"] > 0
    assert (result.run_dir / "source_metadata.json").exists()

