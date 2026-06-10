from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bciworkbench.experiment import Experiment
from bciworkbench.ontology.schemas import ConfigError, parse_experiment_spec
from bciworkbench.sources.xdf import XDFReplaySource


def _write_xdf_json_fixture(path: Path) -> None:
    sfreq = 50.0
    duration_s = 30.0
    n_samples = int(sfreq * duration_s)
    absolute_start = 1000.0
    times = absolute_start + np.arange(n_samples) / sfreq
    relative = times - absolute_start
    data = np.column_stack(
        [
            1e-6 * np.sin(2 * np.pi * 10 * relative),
            1e-6 * np.sin(2 * np.pi * 12 * relative),
            1e-6 * np.sin(2 * np.pi * 20 * relative),
            1e-6 * np.sin(2 * np.pi * 22 * relative),
        ]
    )
    onsets = [absolute_start + value for value in [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0]]
    payload = {
        "streams": [
            {
                "name": "SyntheticEEG",
                "type": "EEG",
                "nominal_srate": sfreq,
                "channel_names": ["C3", "C4", "P3", "P4"],
                "channel_types": ["eeg", "eeg", "eeg", "eeg"],
                "channel_units": ["V", "V", "V", "V"],
                "time_stamps": times.tolist(),
                "time_series": data.tolist(),
            },
            {
                "name": "Markers",
                "type": "Markers",
                "time_stamps": onsets,
                "time_series": [["left"], ["right"], ["left"], ["right"], ["left"], ["right"], ["left"], ["right"]],
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_xdf_replay_source_preserves_marker_time_and_replay_trace(tmp_path: Path) -> None:
    fixture = tmp_path / "recording.json"
    _write_xdf_json_fixture(fixture)

    packet = XDFReplaySource.from_params(
        {"path": str(fixture), "speed_mode": "scaled", "speed": 2.0, "chunk_duration_s": 0.5}
    ).read()

    assert packet.source_id == "xdf_replay"
    assert packet.clock_domain == "recording_clock"
    assert packet.timestamps[0] == 0.0
    assert packet.events[0].onset == 1.0
    assert packet.events[0].metadata["original_marker_time_s"] == 1001.0
    assert packet.metadata["stream_health"]["packet_count"] > 0
    assert packet.metadata["latency_trace"][1]["scheduled_arrival_s"] == 0.25


def test_xdf_replay_source_runs_existing_pipeline_and_writes_latency_artifacts(tmp_path: Path) -> None:
    fixture = tmp_path / "recording.json"
    _write_xdf_json_fixture(fixture)
    config = {
        "schema_version": "0.1",
        "name": "xdf_replay_pipeline",
        "paradigm": "motor_imagery",
        "mode": "replay",
        "random_seed": 1,
        "output_dir": str(tmp_path / "runs"),
        "source": {
            "type": "xdf_replay",
            "path": str(fixture),
            "speed_mode": "fastest",
            "chunk_duration_s": 0.5,
        },
        "pipeline": [
            {"type": "window", "length_s": 1.0, "offset_s": 0.1},
            {"type": "bandpower"},
            {"type": "decoder", "estimator": "nearest_centroid", "calibration_fraction": 0.5},
        ],
        "task": {"type": "motor_imagery_classification", "classes": ["left", "right"]},
    }

    result = Experiment(parse_experiment_spec(config)).run()

    assert result.metrics["source"] == "xdf_replay"
    assert result.metrics["n_predictions"] > 0
    assert (result.run_dir / "latency_trace.csv").exists()
    assert (result.run_dir / "latency_trace.json").exists()
    assert (result.run_dir / "stream_health.json").exists()
    report = (result.run_dir / "report.html").read_text(encoding="utf-8")
    assert "Replay Stream Health" in report


def test_xdf_replay_config_validates_speed_mode() -> None:
    with pytest.raises(ConfigError, match="speed_mode"):
        parse_experiment_spec(
            {
                "name": "bad-xdf",
                "paradigm": "motor_imagery",
                "mode": "replay",
                "source": {"type": "xdf_replay", "path": "missing.xdf", "speed_mode": "warp"},
                "pipeline": [{"type": "window"}, {"type": "bandpower"}, {"type": "decoder"}],
                "task": {"type": "motor_imagery_classification"},
            }
        )
