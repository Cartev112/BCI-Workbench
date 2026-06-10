from __future__ import annotations

from pathlib import Path

import pytest

from bciworkbench.ontology.schema_export import experiment_json_schema
from bciworkbench.ontology.schemas import ConfigError, load_experiment_spec, parse_experiment_spec


def test_example_config_validates() -> None:
    spec = load_experiment_spec(Path("examples/mi_synthetic.yml"))
    assert spec.name == "mi_synthetic_baseline"
    assert spec.source.type == "synthetic_motor_imagery"
    assert [step.type for step in spec.pipeline] == ["window", "bandpower", "decoder"]


def test_invalid_source_is_rejected() -> None:
    with pytest.raises(ConfigError, match="synthetic_motor_imagery"):
        parse_experiment_spec(
            {
                "name": "bad",
                "paradigm": "motor_imagery",
                "source": {"type": "unknown"},
                "pipeline": [{"type": "window"}, {"type": "bandpower"}, {"type": "decoder"}],
                "task": {"type": "motor_imagery_classification"},
            }
        )


def test_invalid_nested_source_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="source.subject"):
        parse_experiment_spec(
            {
                "name": "bad-profile",
                "paradigm": "motor_imagery",
                "source": {
                    "type": "synthetic_motor_imagery",
                    "subject": {"alpha_peak_hz": 10.0, "unknown": 1.0},
                },
                "pipeline": [{"type": "window"}, {"type": "bandpower"}, {"type": "decoder"}],
                "task": {"type": "motor_imagery_classification"},
            }
        )


def test_motor_imagery_rejects_p300_only_source_key() -> None:
    with pytest.raises(ConfigError, match="target_probability"):
        parse_experiment_spec(
            {
                "name": "bad-mi-source",
                "paradigm": "motor_imagery",
                "source": {"type": "synthetic_motor_imagery", "target_probability": 0.25},
                "pipeline": [{"type": "window"}, {"type": "bandpower"}, {"type": "decoder"}],
                "task": {"type": "motor_imagery_classification"},
            }
        )


def test_short_pipeline_is_rejected_cleanly() -> None:
    with pytest.raises(ConfigError, match="pipeline order"):
        parse_experiment_spec(
            {
                "name": "bad-pipeline",
                "paradigm": "motor_imagery",
                "source": {"type": "synthetic_motor_imagery"},
                "pipeline": [{"type": "window"}, {"type": "decoder"}],
                "task": {"type": "motor_imagery_classification"},
            }
        )


def test_invalid_decoder_estimator_is_rejected() -> None:
    with pytest.raises(ConfigError, match="decoder estimator"):
        parse_experiment_spec(
            {
                "name": "bad-decoder",
                "paradigm": "motor_imagery",
                "source": {"type": "synthetic_motor_imagery"},
                "pipeline": [
                    {"type": "window"},
                    {"type": "bandpower"},
                    {"type": "decoder", "estimator": "unknown"},
                ],
                "task": {"type": "motor_imagery_classification"},
            }
        )


def test_experiment_json_schema_exports_required_config_shape() -> None:
    schema = experiment_json_schema()
    assert schema["required"] == ["name", "paradigm", "source", "pipeline", "task"]
    assert "subject" in schema["properties"]["source"]["properties"]
    assert "mne_raw" in schema["properties"]["source"]["properties"]["type"]["enum"]
    assert "moabb" in schema["properties"]["source"]["properties"]["type"]["enum"]
    assert "synthetic_p300" in schema["properties"]["source"]["properties"]["type"]["enum"]
    assert "xdf_replay" in schema["properties"]["source"]["properties"]["type"]["enum"]
    assert "cursor_1d" in schema["properties"]["task"]["properties"]["type"]["enum"]
    assert "confidence_gated" in schema["properties"]["adaptation"]["properties"]["type"]["enum"]
