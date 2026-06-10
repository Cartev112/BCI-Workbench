from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bciworkbench.cli.main import main


def _write_replay_fixture(path: Path) -> None:
    sfreq = 50.0
    n_samples = int(sfreq * 20.0)
    times = np.arange(n_samples) / sfreq
    data = np.column_stack(
        [
            1e-6 * np.sin(2 * np.pi * 10 * times),
            1e-6 * np.sin(2 * np.pi * 12 * times),
        ]
    )
    payload = {
        "streams": [
            {
                "name": "EEG",
                "type": "EEG",
                "nominal_srate": sfreq,
                "channel_names": ["C3", "C4"],
                "time_stamps": times.tolist(),
                "time_series": data.tolist(),
            },
            {
                "name": "Markers",
                "type": "Markers",
                "time_stamps": [1.0, 4.0, 7.0, 10.0, 13.0, 16.0],
                "time_series": [["left"], ["right"], ["left"], ["right"], ["left"], ["right"]],
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_replay_cli_runs_replay_config_with_speed_override(tmp_path: Path, capsys) -> None:
    fixture = tmp_path / "recording.json"
    _write_replay_fixture(fixture)
    config = tmp_path / "replay.yml"
    config.write_text(
        "\n".join(
            [
                'schema_version: "0.1"',
                "name: cli_replay",
                "paradigm: motor_imagery",
                "mode: replay",
                f"output_dir: {tmp_path / 'runs'}",
                "source:",
                "  type: xdf_replay",
                f"  path: {fixture}",
                "  speed_mode: fastest",
                "  chunk_duration_s: 0.5",
                "pipeline:",
                "  - type: window",
                "    length_s: 1.0",
                "    offset_s: 0.1",
                "  - type: bandpower",
                "  - type: decoder",
                "    estimator: nearest_centroid",
                "    calibration_fraction: 0.5",
                "task:",
                "  type: motor_imagery_classification",
                "  classes: [left, right]",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["replay", str(config), "--speed-mode", "scaled", "--speed", "2.0"]) == 0
    output = capsys.readouterr().out
    assert "stream_health:" in output
    assert "latency_trace:" in output
