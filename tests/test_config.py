from __future__ import annotations

from pathlib import Path

import pytest

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

