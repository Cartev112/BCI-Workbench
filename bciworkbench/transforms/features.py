from __future__ import annotations

import numpy as np

from bciworkbench.ontology.packets import FeaturePacket, WindowPacket


class BandpowerTransform:
    def __init__(self, bands: tuple[tuple[str, float, float], ...] | None = None) -> None:
        self.bands = bands or (("mu", 8.0, 13.0), ("beta", 13.0, 30.0))

    @classmethod
    def from_params(cls, params: dict) -> "BandpowerTransform":
        raw_bands = params.get("bands")
        if raw_bands is None:
            return cls()
        bands = tuple((str(item["name"]), float(item["fmin"]), float(item["fmax"])) for item in raw_bands)
        return cls(bands=bands)

    def transform(self, windows: list[WindowPacket], sampling_rate: float) -> list[FeaturePacket]:
        features: list[FeaturePacket] = []
        for window in windows:
            freqs = np.fft.rfftfreq(window.data.shape[1], d=1.0 / sampling_rate)
            spectrum = np.abs(np.fft.rfft(window.data, axis=1)) ** 2
            values: list[float] = []
            names: list[str] = []
            for channel_index in range(window.data.shape[0]):
                for band_name, fmin, fmax in self.bands:
                    mask = (freqs >= fmin) & (freqs < fmax)
                    power = float(np.mean(spectrum[channel_index, mask])) if np.any(mask) else 0.0
                    values.append(np.log10(power + 1e-30))
                    names.append(f"ch{channel_index + 1}_{band_name}")
            features.append(
                FeaturePacket(
                    feature_id=f"features-{window.window_id}",
                    features=np.asarray(values, dtype=float),
                    feature_names=tuple(names),
                    window_id=window.window_id,
                    label=window.label,
                )
            )
        return features

