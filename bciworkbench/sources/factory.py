from __future__ import annotations

from bciworkbench.ontology.schemas import SourceSpec
from bciworkbench.sources.mne import MNERawSource
from bciworkbench.sources.moabb import MOABBSource
from bciworkbench.sources.synthetic import SyntheticMotorImagerySource


def build_source(source: SourceSpec, seed: int):
    if source.type == "synthetic_motor_imagery":
        return SyntheticMotorImagerySource.from_params(source.params, seed=seed)
    if source.type == "mne_raw":
        return MNERawSource.from_params(source.params)
    if source.type == "moabb":
        return MOABBSource.from_params(source.params)
    raise ValueError(f"unsupported source type: {source.type}")

