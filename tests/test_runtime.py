from __future__ import annotations

from pathlib import Path

from bciworkbench.experiment import Experiment


def test_runtime_writes_graph_and_telemetry(tmp_path: Path) -> None:
    experiment = Experiment.from_yaml("examples/mi_synthetic.yml")
    spec = experiment.spec
    patched_spec = type(spec)(
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

    result = Experiment(patched_spec).run()

    graph = result.run_dir / "graph.json"
    telemetry = result.run_dir / "telemetry.jsonl"
    assert graph.exists()
    assert telemetry.exists()
    rows = telemetry.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 4
    assert "source.synthetic_motor_imagery" in rows[0]
