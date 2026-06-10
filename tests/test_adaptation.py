from __future__ import annotations

import json
from pathlib import Path

import pytest

from bciworkbench.adaptation.factory import build_adaptation_adapter
from bciworkbench.experiment import Experiment
from bciworkbench.ontology.packets import IntentPacket
from bciworkbench.ontology.schemas import ConfigError, parse_experiment_spec


def _prediction(index: int, label: str, intent: str, confidence: float = 0.9) -> IntentPacket:
    return IntentPacket(
        intent_id=f"prediction-{index:04d}",
        intent=intent,
        confidence=confidence,
        posterior={intent: confidence},
        latency_ms=1.0,
        window_id=f"window-{index:04d}",
        decoder_id="test_decoder",
        label=label,
    )


def test_supervised_batch_recalibration_corrects_future_chunks() -> None:
    predictions = [
        _prediction(0, "left", "right"),
        _prediction(1, "left", "right"),
        _prediction(2, "right", "left"),
        _prediction(3, "right", "left"),
        _prediction(4, "left", "right"),
        _prediction(5, "right", "left"),
    ]
    adapter = build_adaptation_adapter({"type": "supervised_batch", "batch_size": 4, "min_samples": 2})

    result = adapter.adapt(predictions)

    assert len(result.packets) == 1
    assert [prediction.intent for prediction in result.predictions[:4]] == ["right", "right", "left", "left"]
    assert [prediction.intent for prediction in result.predictions[4:]] == ["left", "right"]
    assert result.metrics["adaptation_changed_prediction_count"] == 2
    assert result.metrics["adaptation_balanced_accuracy_after"] > result.metrics["adaptation_balanced_accuracy_before"]


def test_confidence_gated_recalibration_skips_low_confidence_updates() -> None:
    predictions = [
        _prediction(0, "left", "right", confidence=0.4),
        _prediction(1, "left", "right", confidence=0.4),
        _prediction(2, "right", "left", confidence=0.4),
        _prediction(3, "right", "left", confidence=0.4),
    ]
    adapter = build_adaptation_adapter({"type": "confidence_gated", "confidence_gate": 0.8, "batch_size": 4})

    result = adapter.adapt(predictions)

    assert result.packets == []
    assert [prediction.intent for prediction in result.predictions] == ["right", "right", "left", "left"]
    assert result.metrics["adaptation_update_count"] == 0


def test_adaptation_config_validation_rejects_bad_gate() -> None:
    with pytest.raises(ConfigError, match="confidence_gate"):
        parse_experiment_spec(
            {
                "name": "bad-adaptation",
                "paradigm": "motor_imagery",
                "source": {"type": "synthetic_motor_imagery"},
                "pipeline": [{"type": "window"}, {"type": "bandpower"}, {"type": "decoder"}],
                "task": {"type": "motor_imagery_classification"},
                "adaptation": {"type": "confidence_gated", "confidence_gate": 2.0},
            }
        )


def test_experiment_writes_adaptation_artifacts(tmp_path: Path) -> None:
    experiment = Experiment.from_yaml("examples/mi_synthetic.yml")
    spec = experiment.spec
    patched_spec = type(spec)(
        schema_version=spec.schema_version,
        name="mi_adaptation_noop",
        paradigm=spec.paradigm,
        mode=spec.mode,
        source=spec.source,
        pipeline=spec.pipeline,
        task=spec.task,
        adaptation=type(spec.adaptation)(type="noop", params={}),
        metrics=spec.metrics,
        random_seed=spec.random_seed,
        output_dir=str(tmp_path),
        metadata=spec.metadata,
    )

    result = Experiment(patched_spec).run()

    assert result.metrics["adaptation_update_count"] == 0
    assert (result.run_dir / "adaptation_metrics.json").exists()
    assert (result.run_dir / "adaptation.jsonl").exists()
    assert (result.run_dir / "predictions_before_adaptation.csv").exists()
    metrics = json.loads((result.run_dir / "adaptation_metrics.json").read_text(encoding="utf-8"))
    assert metrics["adaptation_changed_prediction_count"] == 0
    report = (result.run_dir / "report.html").read_text(encoding="utf-8")
    assert "Adaptation" in report
