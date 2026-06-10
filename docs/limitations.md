# Non-Goals And Limitations

BCI Workbench is currently an architecture and benchmarking workbench. It is not
a validated clinical, diagnostic, or therapeutic system.

## Current Non-Goals

- No clinical claims.
- No medical-device safety guarantees.
- No real-time live hardware support yet; Phase 9 LSL/BrainFlow work was
  intentionally skipped for now.
- No arbitrary DAG scheduler; the runtime is still a deterministic linear graph.
- No validated head model or source-space EEG simulation.
- No production online adaptation algorithm.
- No notebook suite yet; CLI examples are the stable surface.

## Simulation Limits

Synthetic MI and P300 sources are useful for controlled software tests and
stress sweeps, but they should not be treated as human physiology. Stressors are
explicit and auditable, not proof of realism.

Closed-loop `cursor_1d` metrics evaluate a simple task model driven by decoder
predictions. They are useful for exposing gaps between classification accuracy
and task success, not for predicting a user's real BCI performance.

## Optional Dependencies

Optional adapters are guarded:

- `bciworkbench[mne]` for MNE Raw FIF.
- `bciworkbench[moabb]` for MOABB datasets.
- `bciworkbench[pyriemann]` for pyRiemann MDM.
- `bciworkbench[xdf]` for real `.xdf` files via `pyxdf`.

Tests avoid network-heavy or hardware-heavy dependencies by using guards and
small fixtures where possible.

## Stability Expectations

The repo is still pre-1.0. Config names and artifact fields should be treated as
milestone contracts, not permanent public API guarantees. When adding features,
prefer preserving existing artifacts and adding new files/fields over changing
the meaning of current ones.
