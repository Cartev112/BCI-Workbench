from __future__ import annotations

from pathlib import Path

import json

import pytest

from bciworkbench.stressbench import (
    BUILTIN_PRESETS,
    aggregate_rows,
    load_stressbench_spec,
    robustness_summary,
    run_stressbench,
)


def test_builtin_presets_include_core_stressors() -> None:
    assert "clean" in BUILTIN_PRESETS
    assert "low_snr" in BUILTIN_PRESETS
    assert "session_drift" in BUILTIN_PRESETS
    assert "delayed_feedback" in BUILTIN_PRESETS


def test_load_stressbench_spec_resolves_relative_base_config() -> None:
    spec = load_stressbench_spec("examples/stressbench_mi.yml")
    assert spec.name == "mi_stressbench"
    assert spec.base_config.name == "mi_synthetic.yml"
    assert "low_snr" in spec.presets


def test_run_stressbench_writes_summary(tmp_path: Path) -> None:
    stressbench_config = tmp_path / "stressbench.yml"
    stressbench_config.write_text(
        "\n".join(
            [
                "name: test_stressbench",
                f"base_config: {Path('examples/mi_synthetic.yml').resolve()}",
                f"output_dir: {tmp_path}",
                "repeats: 1",
                "presets:",
                "  - clean",
                "  - low_snr",
            ]
        ),
        encoding="utf-8",
    )

    result = run_stressbench(stressbench_config)

    assert len(result.rows) == 2
    assert len(result.aggregates) == 2
    assert result.robustness["robustness_score"] is not None
    assert (result.summary_dir / "stressbench_summary.json").exists()
    assert (result.summary_dir / "stressbench_summary.csv").exists()
    assert (result.summary_dir / "stressbench_aggregates.csv").exists()
    assert (result.summary_dir / "stressbench_report.html").exists()
    payload = json.loads((result.summary_dir / "stressbench_summary.json").read_text(encoding="utf-8"))
    assert "aggregates" in payload
    assert "robustness" in payload
    aggregate_by_preset = {row["preset"]: row for row in payload["aggregates"]}
    assert aggregate_by_preset["low_snr"]["balanced_accuracy_mean"] <= aggregate_by_preset["clean"]["balanced_accuracy_mean"]


def test_aggregate_rows_scores_against_clean() -> None:
    rows = [
        {"preset": "clean", "description": "clean", "accuracy": 1.0, "balanced_accuracy": 1.0},
        {"preset": "clean", "description": "clean", "accuracy": 0.9, "balanced_accuracy": 0.9},
        {"preset": "low_snr", "description": "low", "accuracy": 0.6, "balanced_accuracy": 0.6},
    ]
    aggregates = aggregate_rows(rows)
    low_snr = next(row for row in aggregates if row["preset"] == "low_snr")
    summary = robustness_summary(aggregates)
    assert low_snr["delta_from_clean"] == pytest.approx(-0.35)
    assert summary["weakest_preset"] == "low_snr"
