from __future__ import annotations

import json
from pathlib import Path

from bciworkbench.experiment import Experiment
from bciworkbench.reports import compare_runs, summarize_run


def _patched_experiment(output_dir: Path, name_suffix: str = "") -> Experiment:
    experiment = Experiment.from_yaml("examples/mi_synthetic.yml")
    spec = experiment.spec
    patched_spec = type(spec)(
        schema_version=spec.schema_version,
        name=f"{spec.name}{name_suffix}",
        paradigm=spec.paradigm,
        mode=spec.mode,
        source=spec.source,
        pipeline=spec.pipeline,
        task=spec.task,
        metrics=spec.metrics,
        random_seed=spec.random_seed,
        output_dir=str(output_dir),
        metadata=spec.metadata,
    )
    return Experiment(patched_spec)


def test_run_report_contains_phase_6_sections(tmp_path: Path) -> None:
    result = _patched_experiment(tmp_path).run()
    report = (result.run_dir / "report.html").read_text(encoding="utf-8")
    assert "Source" in report
    assert "Model Card" in report
    assert "Latency And Runtime" in report
    assert "Graph" in report
    assert "Provenance" in report


def test_summarize_run_reads_saved_artifacts(tmp_path: Path) -> None:
    result = _patched_experiment(tmp_path).run()
    summary = summarize_run(result.run_dir)
    assert summary["run_id"] == result.run_dir.name
    assert summary["source"] == "synthetic_motor_imagery"
    assert summary["decoder"] in {"sklearn_lda", "nearest_centroid_fallback"}
    assert summary["runtime_total_duration_ms"] is not None


def test_compare_runs_writes_json_csv_and_html(tmp_path: Path) -> None:
    first = _patched_experiment(tmp_path, "_a").run()
    second = _patched_experiment(tmp_path, "_b").run()

    comparison = compare_runs([first.run_dir, second.run_dir], output_dir=tmp_path)
    comparison_dir = comparison["comparison_dir"]

    assert comparison["run_count"] == 2
    assert comparison["best_run"]["run_id"] in {first.run_id, second.run_id}
    assert (comparison_dir / "comparison_summary.json").exists()
    assert (comparison_dir / "comparison_summary.csv").exists()
    assert (comparison_dir / "comparison_report.html").exists()
    payload = json.loads((comparison_dir / "comparison_summary.json").read_text(encoding="utf-8"))
    assert payload["run_count"] == 2
