from __future__ import annotations

from pathlib import Path

from bciworkbench.experiment import Experiment


def test_experiment_run_writes_artifacts(tmp_path: Path) -> None:
    experiment = Experiment.from_yaml("examples/mi_synthetic.yml")
    spec = experiment.spec
    spec = type(spec)(
        schema_version=spec.schema_version,
        name=spec.name,
        paradigm=spec.paradigm,
        mode=spec.mode,
        source=spec.source,
        pipeline=spec.pipeline,
        task=spec.task,
        metrics=spec.metrics,
        random_seed=spec.random_seed,
        output_dir=str(tmp_path),
        metadata=spec.metadata,
    )
    result = Experiment(spec).run()
    assert result.run_dir.exists()
    assert (result.run_dir / "metrics.json").exists()
    assert (result.run_dir / "source_metadata.json").exists()
    assert (result.run_dir / "events.csv").exists()
    assert (result.run_dir / "windows.csv").exists()
    assert (result.run_dir / "predictions.csv").exists()
    assert (result.run_dir / "report.html").exists()
    assert result.metrics["n_predictions"] > 0
