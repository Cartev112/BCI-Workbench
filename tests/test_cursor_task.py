from __future__ import annotations

import json
from pathlib import Path

import pytest

from bciworkbench.experiment import Experiment
from bciworkbench.ontology.packets import IntentPacket
from bciworkbench.ontology.schemas import ConfigError, parse_experiment_spec
from bciworkbench.tasks.cursor import run_cursor_task


def _prediction(index: int, label: str, intent: str | None = None, confidence: float = 1.0) -> IntentPacket:
    predicted = intent or label
    return IntentPacket(
        intent_id=f"prediction-{index:04d}",
        intent=predicted,
        confidence=confidence,
        posterior={predicted: confidence},
        latency_ms=1.0,
        window_id=f"window-{index:04d}",
        decoder_id="test_decoder",
        label=label,
    )


def test_cursor_task_success_metrics_without_delay() -> None:
    predictions = [_prediction(index, "left" if index % 2 == 0 else "right") for index in range(6)]
    result = run_cursor_task(predictions, {"feedback_delay_ms": 0.0})
    assert result.metrics["target_acquisition_rate"] == 1.0
    assert result.metrics["mean_time_to_target_s"] == 0.5
    assert result.metrics["false_activation_rate"] == 0.0
    assert len(result.feedback) == len(predictions)


def test_cursor_task_feedback_delay_reduces_target_acquisition() -> None:
    predictions = [_prediction(index, "left" if index % 2 == 0 else "right") for index in range(8)]
    no_delay = run_cursor_task(predictions, {"feedback_delay_ms": 0.0})
    delayed = run_cursor_task(predictions, {"feedback_delay_ms": 500.0, "control_interval_s": 0.5})
    assert delayed.metrics["target_acquisition_rate"] < no_delay.metrics["target_acquisition_rate"]
    assert delayed.metrics["false_activation_rate"] > no_delay.metrics["false_activation_rate"]


def test_cursor_task_config_validation_rejects_negative_delay() -> None:
    with pytest.raises(ConfigError, match="feedback_delay_ms"):
        parse_experiment_spec(
            {
                "name": "bad-cursor",
                "paradigm": "motor_imagery",
                "source": {"type": "synthetic_motor_imagery"},
                "pipeline": [{"type": "window"}, {"type": "bandpower"}, {"type": "decoder"}],
                "task": {"type": "cursor_1d", "feedback_delay_ms": -1},
            }
        )


def test_cursor_example_runs_and_writes_task_artifacts(tmp_path: Path) -> None:
    experiment = Experiment.from_yaml("examples/mi_cursor_synthetic.yml")
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
    assert result.metrics["source"] == "synthetic_motor_imagery"
    assert result.metrics["target_acquisition_rate"] is not None
    assert "decoder_task_gap" in result.metrics
    assert (result.run_dir / "task_metrics.json").exists()
    assert (result.run_dir / "task_states.csv").exists()
    assert (result.run_dir / "feedback.csv").exists()
    task_metrics = json.loads((result.run_dir / "task_metrics.json").read_text(encoding="utf-8"))
    assert task_metrics["feedback_delay_ms"] == 80.0
    report = (result.run_dir / "report.html").read_text(encoding="utf-8")
    assert "Closed Loop Task" in report
