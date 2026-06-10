from __future__ import annotations

import numpy as np

from bciworkbench.decoders.sklearn import SklearnDecoder
from bciworkbench.ontology.packets import FeaturePacket


def _features() -> list[FeaturePacket]:
    rows = [
        ("left", [0.0, 0.1]),
        ("right", [2.0, 2.1]),
        ("left", [0.2, 0.0]),
        ("right", [2.1, 2.0]),
        ("left", [0.1, 0.2]),
        ("right", [2.2, 2.1]),
        ("left", [0.0, 0.3]),
        ("right", [2.3, 2.2]),
    ]
    return [
        FeaturePacket(
            feature_id=f"f{index}",
            features=np.asarray(values, dtype=float),
            feature_names=("a", "b"),
            window_id=f"w{index}",
            label=label,
        )
        for index, (label, values) in enumerate(rows)
    ]


def test_sklearn_decoder_nearest_centroid_fit_predict_and_card() -> None:
    decoder = SklearnDecoder(estimator="nearest_centroid", calibration_fraction=0.5)
    result = decoder.fit_predict(_features())
    assert result.decoder_name == "nearest_centroid"
    assert result.train_size == 4
    assert result.test_size == 4
    assert result.calibration_time_s >= 0
    assert result.model_card["feature_count"] == 2


def test_sklearn_decoder_logistic_recipe_runs() -> None:
    decoder = SklearnDecoder(estimator="logistic_regression", calibration_fraction=0.5)
    result = decoder.fit_predict(_features())
    assert result.decoder_name in {"sklearn_logistic_regression", "nearest_centroid_fallback"}
    assert len(result.predictions) == 4


def test_sklearn_decoder_save_and_load(tmp_path) -> None:
    decoder = SklearnDecoder(estimator="nearest_centroid", calibration_fraction=0.5)
    decoder.fit_predict(_features())
    model_path = tmp_path / "decoder.pkl"
    decoder.save(model_path)
    loaded = SklearnDecoder.load(model_path)
    assert loaded.decoder_name == "nearest_centroid"
    assert loaded.classes_ == ["left", "right"]
