# Source Adapter Guide

Source adapters convert signals and events into `SignalPacket`s. The downstream
pipeline should not need to know whether data came from a generator, file, public
dataset, or replay source.

## Implemented Sources

| Source type | Mode | Dependency | Notes |
| --- | --- | --- | --- |
| `synthetic_motor_imagery` | synthetic | core | Deterministic MI-like EEG with trial/cue events and configurable stressors. |
| `synthetic_p300` | synthetic | core | Deterministic oddball source with target/non-target stimulus events. |
| `mne_raw` | offline | `bciworkbench[mne]` | Reads MNE Raw FIF and converts annotations to events. |
| `moabb` | offline | `bciworkbench[moabb]` | Initial BNCI2014_001 adapter through MOABB. |
| `xdf_replay` | replay | `bciworkbench[xdf]` for `.xdf`; JSON fixtures use core | Reads signal and marker streams, then adds replay timing telemetry. |

Optional adapters fail with install hints instead of importing heavy packages at
core import time.

## Replay Source

`xdf_replay` supports:

- `speed_mode`: `fastest`, `real_time`, `scaled`, or `stepped`.
- `speed`: scale factor for `scaled` mode.
- `chunk_duration_s`: replay packet size.
- `step_duration_s`: fixed step duration for stepped mode.
- `processing_time_ms`: deterministic processing cost used to model backlog.
- `queue_capacity`: optional queue-depth threshold for dropped packet telemetry.

Replay runs write:

- `latency_trace.csv`
- `latency_trace.json`
- `stream_health.json`

Signal timestamps and marker timestamps remain recording-relative. Packet arrival
times are replay telemetry, not replacements for sample timestamps.

## Adding A Source

A source should expose `read() -> SignalPacket`, preserve source-specific
metadata, and map external labels into typed `Event`s. Use explicit conversion
code rather than string-only heuristics where the upstream library exposes
structured metadata.

Minimum expectations:

- Fill `ChannelSchema` names, types, units, sampling rate, reference, and bad
  channels where known.
- Preserve original source identifiers and stream/session metadata.
- Keep optional dependencies guarded with clear install hints.
- Add a test that constructs the adapter without network or hardware when
  possible.
