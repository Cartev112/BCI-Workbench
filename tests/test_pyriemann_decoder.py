from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

from bciworkbench.decoders.pyriemann import PyRiemannDecoder
from bciworkbench.ontology.packets import FeaturePacket


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

