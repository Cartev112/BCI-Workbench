from __future__ import annotations

from bciworkbench.ontology.packets import SignalPacket, WindowPacket


class TrialWindowTransform:
    def __init__(self, length_s: float = 1.5, offset_s: float = 0.3) -> None:
        if length_s <= 0:
            raise ValueError("length_s must be positive")
        if offset_s < 0:
            raise ValueError("offset_s must be non-negative")
        self.length_s = length_s
        self.offset_s = offset_s

    @classmethod
    def from_params(cls, params: dict) -> "TrialWindowTransform":
        return cls(length_s=float(params.get("length_s", 1.5)), offset_s=float(params.get("offset_s", 0.3)))

    def transform(self, packet: SignalPacket) -> list[WindowPacket]:
        windows: list[WindowPacket] = []
        sr = packet.channel_schema.sampling_rate
        trial_events = [event for event in packet.events if event.event_type == "trial.start"]
        for index, event in enumerate(trial_events):
            start_time = event.onset + self.offset_s
            end_time = start_time + self.length_s
            sample_start = int(round(start_time * sr))
            sample_end = int(round(end_time * sr))
            if sample_start < 0 or sample_end > packet.data.shape[1] or sample_end <= sample_start:
                continue
            related = tuple(
                item
                for item in packet.events
                if item.onset >= event.onset and item.onset <= event.onset + max(event.duration, self.length_s)
            )
            windows.append(
                WindowPacket(
                    window_id=f"window-{index:04d}",
                    data=packet.data[:, sample_start:sample_end].copy(),
                    start_time=start_time,
                    end_time=end_time,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    label=event.target,
                    events=related,
                    metadata={"source_event_id": event.event_id},
                )
            )
        return windows

