# Ontology Guide

BCI Workbench keeps data moving through explicit packet types so synthetic,
offline, replay, and later live runs can be compared without hiding timing or
provenance.

## Core Packets

- `SignalPacket`: channel x sample signal data, sample timestamps, channel schema,
  events, clock domain, source id, quality fields, and source metadata.
- `Event`: typed event with onset, duration, sample index, clock domain, target,
  source, and metadata. Trial events use `event_type: trial.start`.
- `WindowPacket`: a view over a signal packet aligned to a trial event.
- `FeaturePacket`: one feature vector for one window.
- `IntentPacket`: decoder output with intent, confidence, posterior, latency,
  window id, decoder id, and optional true label.
- `TaskStatePacket`: closed-loop task state, observation, target, reward, done,
  and success fields.
- `FeedbackPacket`: feedback/action record with render time, delay, reward, and
  the task state it came from.
- `AdaptationPacket`: decoder or policy update record with input window ids,
  labels, parameters changed, and before/after metrics.

## Clock Domains

Supported clock domains are defined in `bciworkbench.ontology.timing`.

- `sample_clock`: synthetic or sample-derived timing.
- `recording_clock`: timestamps from offline recordings or replay files.
- `lsl_clock`: Lab Streaming Layer local clock, reserved for live/replay parity.
- `system_clock`: host wall-clock time.
- `sim_clock`: closed-loop task or simulation time.

Replay sources preserve original marker timing separately from simulated packet
arrival timing. Source metadata and `latency_trace.csv` should be used when
debugging timing behavior.

## Configuration Shape

Experiment configs are YAML mappings with these main blocks:

- `source`: data source adapter and source-specific parameters.
- `pipeline`: currently `window`, then `bandpower`, `erp_features`, or
  `covariance`, then `decoder`.
- `task`: classification task or `cursor_1d`.
- `adaptation`: optional, defaults to `none`.
- `metrics`: requested metrics for report readability.

Generate the current JSON Schema with:

```powershell
bciworkbench schema experiment
```

The schema is an implementation contract for the current milestone, not a
claim that every future BCI task is already covered.
