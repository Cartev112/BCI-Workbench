from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

from bciworkbench.decoders.pyriemann import PyRiemannDecoder
from bciworkbench.graph.nodes import _feature_transform
from bciworkbench.ontology.packets import FeaturePacket
from bciworkbench.ontology.schemas import parse_experiment_spec


def test_pyriemann_decoder_dependency_guard_or_runs() -> None:
    features = [
        FeaturePacket(
            feature_id=f"f{index}",
            features=np.eye(2).reshape(-1) * (index + 1),
            feature_names=("c11", "c12", "c21", "c22"),
            window_id=f"w{index}",
            label="left" if index % 2 == 0 else "right",
        )
        for index in range(6)
    ]
    decoder = PyRiemannDecoder(calibration_fraction=0.5)
    if find_spec("pyriemann") is None:
        with pytest.raises(ImportError, match="bciworkbench\\[pyriemann\\]"):
            decoder.fit_predict(features)
    else:
        result = decoder.fit_predict(features)
        assert result.decoder_name == "pyriemann_mdm"


def test_covariance_feature_transform_outputs_square_features() -> None:
    from bciworkbench.ontology.packets import Event, WindowPacket

    window = WindowPacket(
        window_id="w1",
        data=np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        start_time=0.0,
        end_time=0.03,
        sample_start=0,
        sample_end=3,
        label="left",
        events=(Event(event_id="e1", event_type="trial.start", name="left", onset=0.0),),
    )
    features = _feature_transform("covariance", {}).transform([window], sampling_rate=100.0)
    assert features[0].features.shape == (4,)


def test_pyriemann_pipeline_config_validates() -> None:
    spec = parse_experiment_spec(
        {
            "name": "pyriemann-config",
            "paradigm": "motor_imagery",
            "source": {"type": "synthetic_motor_imagery"},
            "pipeline": [
                {"type": "window"},
                {"type": "covariance"},
                {"type": "decoder", "adapter": "pyriemann", "estimator": "mdm"},
            ],
            "task": {"type": "motor_imagery_classification"},
        }
    )
    assert spec.pipeline[1].type == "covariance"
