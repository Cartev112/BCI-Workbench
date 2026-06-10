from __future__ import annotations

from importlib.util import find_spec

import pytest

from bciworkbench.ontology.schemas import ConfigError, parse_experiment_spec
from bciworkbench.sources.moabb import MOABBSource


def test_moabb_source_config_validates_bnci2014_001() -> None:
    spec = parse_experiment_spec(
        {
            "name": "moabb_bnci",
            "paradigm": "motor_imagery",
            "mode": "offline",
            "source": {"type": "moabb", "dataset": "BNCI2014_001", "subject": 1},
            "pipeline": [{"type": "window"}, {"type": "bandpower"}, {"type": "decoder"}],
            "task": {"type": "motor_imagery_classification"},
        }
    )
    assert spec.source.type == "moabb"
    assert spec.source.params["dataset"] == "BNCI2014_001"


def test_moabb_source_rejects_unsupported_dataset() -> None:
    with pytest.raises(ConfigError, match="BNCI2014_001"):
        parse_experiment_spec(
            {
                "name": "moabb_bad",
                "paradigm": "motor_imagery",
                "mode": "offline",
                "source": {"type": "moabb", "dataset": "UNKNOWN", "subject": 1},
                "pipeline": [{"type": "window"}, {"type": "bandpower"}, {"type": "decoder"}],
                "task": {"type": "motor_imagery_classification"},
            }
        )


def test_moabb_source_constructs_without_downloading() -> None:
    source = MOABBSource.from_params({"dataset": "BNCI2014_001", "subject": 1})
    assert source.config.dataset == "BNCI2014_001"
    assert source.config.subject == 1


def test_moabb_source_dependency_guard_or_available() -> None:
    source = MOABBSource.from_params({"dataset": "BNCI2014_001", "subject": 1})
    if find_spec("moabb") is None:
        with pytest.raises(ImportError, match="bciworkbench\\[moabb\\]"):
            source.read()
    else:
        assert find_spec("moabb") is not None

