from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_CLOCK_DOMAINS = {
    "sample_clock",
    "monotonic_clock",
    "wall_clock",
    "lsl_clock",
    "recording_clock",
    "sim_clock",
}


@dataclass(frozen=True)
class ClockDomain:
    """Named timestamp origin used by packets and events."""

    name: str
    description: str

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_CLOCK_DOMAINS:
            raise ValueError(f"unsupported clock domain: {self.name}")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}


CLOCK_DOMAINS = {
    "sample_clock": ClockDomain("sample_clock", "Timestamps derived from sample index and sampling rate."),
    "monotonic_clock": ClockDomain("monotonic_clock", "Local monotonic process time."),
    "wall_clock": ClockDomain("wall_clock", "Human-readable system time."),
    "lsl_clock": ClockDomain("lsl_clock", "Lab Streaming Layer local clock."),
    "recording_clock": ClockDomain("recording_clock", "Timestamps stored in a recording file."),
    "sim_clock": ClockDomain("sim_clock", "Deterministic simulated time."),
}

