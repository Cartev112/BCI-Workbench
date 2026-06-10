# BCI Workbench Implementation Plan

## 1. Product Thesis

BCI Workbench is a Python-first systems workbench for defining, simulating, replaying, benchmarking, and eventually deploying complete BCI pipelines.

The core promise:

> Define a BCI architecture once, then run it against synthetic subjects, public datasets, recorded streams, replayed real-time streams, and live hardware while measuring accuracy, latency, drift, adaptation, and closed-loop task performance with comparable reports.

The project should not become another signal-processing library, model zoo, acquisition driver, or dataset repository. The ecosystem already has strong component libraries:

- MNE-Python for neurophysiology data structures, preprocessing, and analysis.
- MOABB for public EEG BCI benchmark datasets and offline evaluation.
- Braindecode for deep-learning EEG/ECoG/MEG/iEEG decoding.
- pyRiemann for classical Riemannian BCI pipelines.
- BrainFlow and vendor SDKs for board-level acquisition.
- LSL, MNE-LSL, XDF, BIDS, and NWB for streaming, recordings, and standards.
- Gymnasium and RL libraries for environment and policy APIs.

BCI Workbench should own the missing transition layer:

> The architecture reproducibility layer between offline BCI algorithms and real-time closed-loop BCI systems.

That means the moat is not "we implemented better filters." The moat is:

1. A good BCI ontology and runtime contract.
2. Simulation realism focused on BCI failure modes.
3. High-quality integration with existing libraries without forcing awkward wrappers.
4. Reproducible reports that expose timing, drift, calibration, and closed-loop performance.
5. Benchmark recipes for whole architectures under distribution shift.

## 2. Scope And Non-Goals

### In Scope

- A typed experiment definition for complete BCI systems.
- A deterministic graph runtime for offline and simulated experiments.
- A real-time runtime using the same node and packet contracts.
- Adapters to existing data, DSP, ML, streaming, and task libraries.
- Synthetic BCI data generation with explicit ground truth and stressors.
- Domain-randomized robustness evaluation.
- Closed-loop task environments with latency and feedback modeling.
- Run artifacts, provenance, metrics, and HTML/JSON reports.
- A CLI for running, replaying, validating, comparing, and reporting experiments.

### Out Of Scope

- Replacing MNE, MOABB, Braindecode, pyRiemann, BrainFlow, LSL, BIDS, or NWB.
- Building a large GUI before the API and reports are reliable.
- Claiming physiological realism that is not validated or documented.
- Supporting every modality in v0.1.
- Making RL central to the MVP.
- Inventing a new raw neurodata storage standard.

## 3. Initial Technical Assumptions

- Package name: `bciworkbench`.
- Minimum Python: 3.11.
- Core dependencies should stay light: `numpy`, `scipy`, `pydantic`, `pyyaml`, `rich`, `typer`, `pandas`, and `jinja2` are acceptable candidates.
- Optional integrations must live behind extras:
  - `bciworkbench[mne]`
  - `bciworkbench[moabb]`
  - `bciworkbench[lsl]`
  - `bciworkbench[brainflow]`
  - `bciworkbench[pyriemann]`
  - `bciworkbench[braindecode]`
  - `bciworkbench[gym]`
  - `bciworkbench[all]`
- Config and schema validation should use Pydantic v2.
- High-throughput runtime packets should use dataclasses or small structured classes, not heavy Pydantic validation per packet.
- Public APIs should be typed and stable before implementation breadth expands.

## 4. The Central Design Problem: Ontology First

The hardest part is designing a precise BCI ontology that is broad enough to represent common systems but narrow enough to prevent mushy, untestable abstractions.

The ontology must encode:

- What signal is being measured.
- How channels are named, referenced, located, and transformed.
- What events mean, when they occurred, and which clock they belong to.
- What task state existed when a signal window was decoded.
- What intent or state the decoder inferred.
- What feedback was delivered to the user or simulated subject.
- What adaptation or model update happened.
- What timing, quality, and provenance were attached to each step.

The ontology is the project's foundation. Every source adapter, synthetic generator, task, decoder, report, and benchmark should speak this common language.

### 4.1 Ontology Principles

1. Preserve timing information even when the downstream decoder does not need it.
2. Treat labels, cues, feedback, and user intent as different concepts.
3. Distinguish measured signal, latent simulated ground truth, and decoded intent.
4. Keep extension points explicit through namespaced metadata fields.
5. Avoid hiding clock drift, marker lag, dropped samples, or uncertain labels.
6. Do not make adapters "mostly work" by discarding semantics.
7. Version every schema that can appear in a run artifact.

### 4.2 Core Ontology Objects

#### ExperimentSpec

The portable unit of experimentation.

Fields:

- `schema_version`
- `name`
- `description`
- `paradigm`
- `mode`: `synthetic`, `offline`, `replay`, `live`
- `source`
- `pipeline`
- `task`
- `adaptation`
- `metrics`
- `logging`
- `random_seed`
- `tags`
- `metadata`

#### RunContext

Runtime context shared across nodes.

Fields:

- `run_id`
- `experiment_id`
- `started_at`
- `mode`
- `clock`
- `rng`
- `artifact_dir`
- `dependency_versions`
- `hardware`
- `source_summary`
- `schema_versions`

#### ClockDomain

A clock domain describes where timestamps come from.

Required clock domains:

- `sample_clock`: timestamps derived from sample index and sampling rate.
- `monotonic_clock`: local monotonic process time.
- `wall_clock`: human-readable system time.
- `lsl_clock`: LSL local clock.
- `recording_clock`: timestamps stored in a file.
- `sim_clock`: deterministic simulated time.

Every packet and event must identify its clock domain. Conversions should be explicit and logged.

#### ChannelSchema

The canonical representation of signal channels.

Fields:

- `names`
- `types`: `eeg`, `ecog`, `ieeg`, `emg`, `ecg`, `eog`, `fnirs`, `spike`, `marker`, `aux`
- `units`
- `sampling_rate`
- `reference`
- `montage`
- `coordinate_frame`
- `positions`
- `bad_channels`
- `quality`
- `metadata`

Rules:

- Signal arrays use `channels x samples`.
- Units must be explicit. EEG should use volts internally, with report display in uV where useful.
- Channel order is part of the schema and must be preserved.
- Channel renaming, dropping, referencing, interpolation, and montage changes should emit transform provenance.

#### Event

Events are first-class ontology objects, not loose string labels.

Fields:

- `id`
- `type`
- `name`
- `onset`
- `duration`
- `clock_domain`
- `sample_index`
- `confidence`
- `source`
- `target`
- `metadata`

Initial event types:

- `session.start`
- `session.end`
- `trial.start`
- `trial.end`
- `stimulus.onset`
- `stimulus.offset`
- `cue.onset`
- `cue.offset`
- `target.presented`
- `response.user`
- `response.device`
- `artifact.blink`
- `artifact.saccade`
- `artifact.emg`
- `artifact.dropout`
- `state.attention`
- `state.fatigue`
- `state.strategy`
- `feedback.presented`
- `feedback.reward`
- `decoder.prediction`
- `decoder.update`
- `adaptation.start`
- `adaptation.end`
- `quality.warning`
- `quality.error`

Important distinction:

- `cue` is an instruction or prompt.
- `target` is the intended class or task goal.
- `response` is what the user or system did.
- `intent` is the latent or decoded internal state.
- `feedback` is what the system showed or delivered.

These should not collapse into a single label column.

#### SignalPacket

The main packet for sampled neural or biosignal data.

Fields:

- `data`: `np.ndarray`, shape `channels x samples`
- `timestamps`: `np.ndarray`, shape `samples`
- `clock_domain`
- `channel_schema`
- `sampling_rate`
- `modality`
- `events`
- `quality`
- `source_id`
- `sequence_id`
- `metadata`

#### WindowPacket

Packet emitted after segmentation.

Fields:

- `data`
- `start_time`
- `end_time`
- `center_time`
- `sample_start`
- `sample_end`
- `events`
- `label`
- `context`
- `metadata`

#### FeaturePacket

Packet emitted by feature extractors or embedders.

Fields:

- `features`
- `feature_schema`
- `source_window`
- `events`
- `metadata`

#### IntentPacket

Decoder output.

Fields:

- `intent`
- `posterior`
- `confidence`
- `uncertainty`
- `latency_ms`
- `decoder_id`
- `window_id`
- `task_state_id`
- `metadata`

#### TaskStatePacket

State of a closed-loop task.

Fields:

- `task_id`
- `state`
- `observation`
- `target`
- `reward`
- `done`
- `success`
- `events`
- `metadata`

#### FeedbackPacket

Feedback delivered to a human, simulated user, or environment.

Fields:

- `action`
- `rendered_at`
- `clock_domain`
- `reward`
- `delay_ms`
- `task_state`
- `metadata`

#### AdaptationPacket

Record of decoder, policy, threshold, calibration, or user-model update.

Fields:

- `adapter_id`
- `update_type`
- `input_window_ids`
- `labels`
- `confidence_gate`
- `parameters_changed`
- `metrics_before`
- `metrics_after`
- `metadata`

### 4.3 Ontology File Layout

Implement ontology in:

```text
bciworkbench/
  ontology/
    __init__.py
    schemas.py
    events.py
    channels.py
    packets.py
    timing.py
    validation.py
```

Implementation split:

- Pydantic models for `ExperimentSpec`, config validation, and reportable schemas.
- Dataclasses for runtime packets.
- JSON Schema export for configs and run artifacts.
- Strict schema versioning from the first commit.

### 4.4 Ontology Acceptance Criteria

- A motor imagery dataset from MOABB can be represented without losing cue, trial, target, and session metadata.
- An XDF replay can preserve signal timestamps, marker timestamps, and stream clock metadata.
- A synthetic closed-loop run can store latent subject state separately from decoded intent.
- A live LSL run can log packet arrival time separately from sample timestamp.
- Every report can reconstruct what each decoder prediction was based on.

## 5. Runtime Architecture

The runtime should execute a typed directed graph of BCI nodes.

Canonical graph:

```text
SignalSource
  -> Synchronizer
  -> ChannelMap
  -> Preprocessor
  -> Windowing
  -> FeatureExtractor or Embedder
  -> Decoder
  -> IntentState
  -> Controller or Policy
  -> TaskEnvironment
  -> Feedback
  -> Adaptation
  -> Recorder
  -> Evaluator
```

Not every experiment uses every node, but every experiment should fit this conceptual lifecycle.

### 5.1 Node Contract

Each node should expose:

```python
class Node:
    node_id: str
    input_types: tuple[type, ...]
    output_types: tuple[type, ...]

    def setup(self, context: RunContext) -> None:
        ...

    def process(self, packet, context: RunContext):
        ...

    def teardown(self, context: RunContext) -> None:
        ...
```

Optional interfaces:

- `AsyncNode`
- `BatchNode`
- `TrainableNode`
- `StatefulNode`
- `SerializableNode`
- `TelemetryNode`

### 5.2 Runtime Modes

#### Offline Mode

Deterministic execution over files, public datasets, or generated data.

Requirements:

- Repeatable order.
- Stable random seeds.
- Faster-than-real-time operation.
- Full artifact logging.
- Useful for tests and benchmarks.

#### Replay Mode

Recorded data is replayed using either original timing, scaled timing, or stepped timing.

Requirements:

- `--speed 1.0` for real-time replay.
- `--speed max` for fastest deterministic replay.
- `--step` for debugging packet-by-packet behavior.
- Packet arrival time and original sample time remain separate.

#### Live Mode

Real-time streaming from LSL, BrainFlow, or another source.

Requirements:

- Bounded queues.
- Backpressure behavior.
- Dropped-sample and backlog reporting.
- Latency and jitter traces.
- Graceful shutdown and partial report writing.

#### Synthetic Mode

Deterministic generated signals, events, task state, and latent variables.

Requirements:

- Ground truth output.
- Configurable random seeds.
- Stressor sweeps.
- Same downstream graph as real data.

### 5.3 Scheduler

Initial scheduler design:

- Start with a simple linear graph executor for v0.1.
- Add DAG validation early, but do not implement arbitrary parallel DAG scheduling until linear execution works.
- Add async queues for real-time mode after deterministic offline tests pass.
- Measure node-level latency from the beginning.

Runtime files:

```text
bciworkbench/
  graph/
    __init__.py
    node.py
    spec.py
    builder.py
    runtime.py
    scheduler.py
    clocks.py
    telemetry.py
    errors.py
```

### 5.4 Runtime Acceptance Criteria

- The same `ExperimentSpec` can run with a synthetic source and an offline source by changing only `source`.
- Offline execution is deterministic under a fixed seed.
- The runtime records per-node processing latency.
- A failed node produces a useful error with node id, packet id, and source context.
- A partial run still writes enough artifacts to debug failure.

## 6. Integration Strategy

Integration quality is one of the moats. The project should feel native to the libraries it wraps.

### 6.1 Dependency Boundaries

Rules:

- Core package must not import heavy optional dependencies at module import time.
- Adapters should fail with a clear install hint.
- Each optional adapter gets unit tests with import guards.
- Conversion functions should be explicit and documented.

Example:

```python
from bciworkbench.sources import MOABBSource
```

If MOABB is missing, the error should say:

```text
MOABBSource requires bciworkbench[moabb].
Install with: pip install "bciworkbench[moabb]"
```

### 6.2 Source Adapters

Initial source priority:

1. `SyntheticSource`
2. `MNERawSource`
3. `MOABBSource`
4. `XDFReplaySource`
5. `LSLSource`
6. `BrainFlowSource`

Later source priority:

7. `BIDSReplaySource`
8. `NWBSource`
9. `MneLslSource`
10. `NeuralDataSimulatorSource`

Source adapter contract:

```python
class SignalSource(Node):
    def iter_packets(self, context: RunContext) -> Iterator[SignalPacket]:
        ...
```

Real-time sources can expose:

```python
class LiveSignalSource(SignalSource):
    async def stream_packets(self, context: RunContext) -> AsyncIterator[SignalPacket]:
        ...
```

### 6.3 MNE Integration

MNE should be the canonical bridge for many offline workflows.

Implementation:

- Convert `mne.io.Raw` to `SignalPacket` batches.
- Convert MNE annotations to `Event`.
- Preserve `info`, channel names, sampling rate, reference, montage, bad channels, and measurement date where possible.
- Support MNE transforms as nodes:
  - bandpass
  - notch
  - resample
  - pick channels
  - set reference
  - epoching adapter

Do not hide MNE objects. Provide escape hatches:

- `packet.to_mne_raw()`
- `MNERawSource.raw`
- `MNETransform.fn`

### 6.4 MOABB Integration

MOABB should provide public benchmark access, not become the whole evaluation layer.

Implementation:

- Load datasets and subjects through MOABB.
- Convert paradigm outputs into `SignalPacket`, `WindowPacket`, and `Event`.
- Preserve dataset, subject, session, run, and paradigm metadata.
- Use MOABB evaluation where useful, but BCI Workbench reports should add timing, task, and architecture-level metrics.

MVP target:

- BNCI2014_001 motor imagery.

### 6.5 XDF And LSL Integration

XDF/LSL are critical for recording-to-live parity.

Implementation:

- `XDFReplaySource` reads signal streams and marker streams.
- Preserve stream names, nominal sampling rates, clock offsets, and marker timestamps.
- Replay modes:
  - fastest deterministic
  - real-time
  - scaled
  - stepped
- `LSLSource` emits live `SignalPacket`s with:
  - sample timestamps
  - local arrival timestamps
  - stream info metadata
  - dropped sample indicators where available

### 6.6 BrainFlow Integration

BrainFlow should be a hardware adapter, not a dependency of the core runtime.

Implementation:

- Map BrainFlow board channels to `ChannelSchema`.
- Surface board id, serial port, MAC/IP parameters, and sampling rate.
- Convert board data chunks to `SignalPacket`.
- Expose signal quality warnings where BrainFlow provides useful data.

### 6.7 Decoder Integrations

Initial decoder adapters:

- `SklearnDecoder`
- `PyRiemannDecoder`
- `TorchDecoder`
- `BraindecodeDecoder`

Decoder contract:

```python
class Decoder(Node):
    def fit(self, windows, labels, context: RunContext) -> None:
        ...

    def predict(self, packet, context: RunContext) -> IntentPacket:
        ...

    def update(self, packet, feedback, context: RunContext) -> AdaptationPacket | None:
        ...
```

Decoder lifecycle should include:

- calibration
- prediction
- uncertainty or posterior when available
- online update
- save/load
- model card metadata
- latency measurement
- domain-shift diagnostics

### 6.8 Gymnasium Integration

Closed-loop tasks should optionally be Gymnasium-compatible.

Rule:

> All BCI closed-loop tasks are environments; some can be optimized with RL.

MVP should include Gym compatibility for cursor control, but not depend on RL.

## 7. Simulation Realism Plan

Simulation is the largest technical moat after ontology. The goal is not generic "realistic EEG." The goal is:

> BCI task data with controllable subject, session, artifact, intent, feedback, timing, and adaptation parameters.

Synthetic runs must emit:

- signal
- events
- intended task labels
- latent user state
- artifact labels and masks
- timing truth
- source or component truth when available
- stressor parameters

### 7.1 Simulation Levels

#### Level 0: Plumbing Synthetic

Purpose:

- Test runtime, packet flow, logging, reports, and CLI.

Features:

- Sinusoids.
- White and pink noise.
- Configurable channel count and sampling rate.
- Deterministic event stream.

Acceptance:

- Fast, deterministic, and easy to inspect.
- Not presented as physiologically realistic.

#### Level 1: Spectral EEG-Like Synthetic

Purpose:

- Provide EEG-shaped background data for early pipelines.

Features:

- 1/f background.
- Alpha, mu, beta, theta rhythms.
- Spatial covariance across channels.
- Channel-specific noise.
- Line noise at 50 or 60 Hz.
- Basic EOG and EMG artifacts.

Acceptance:

- PSD resembles broad EEG bands.
- Channel covariance is non-trivial.
- Output remains deterministic under seed.

#### Level 2: Paradigm-Aware Synthetic

Purpose:

- Generate task-specific BCI signals with labels and events.

Initial paradigms:

- Motor imagery left/right.
- P300 oddball/speller.
- SSVEP target selection.

Motor imagery features:

- Mu and beta ERD/ERS.
- Class-specific lateralized topographies.
- Trial-level amplitude variability.
- Cue-to-response timing.
- Subject-specific separability.

P300 features:

- Event-locked positive component around configurable latency.
- Latency jitter.
- Amplitude variability.
- Attention/fatigue modulation.
- Target and non-target event labels.

SSVEP features:

- Target frequency and harmonics.
- Phase variability.
- Occipital topography.
- Frequency-specific SNR.

Acceptance:

- Baseline decoders should perform above chance under clean settings.
- Performance should degrade as SNR, drift, jitter, or artifacts worsen.
- Reports should show ground truth separability settings.

#### Level 3: Subject And Session Model

Purpose:

- Model subject variability and session-to-session distribution shift.

Subject parameters:

- alpha peak frequency
- mu peak frequency
- signal amplitude
- skull/scalp conductivity proxy
- artifact tendency
- attention baseline
- fatigue rate
- motor imagery vividness
- P300 amplitude
- SSVEP response gain
- learning rate
- strategy shift probability

Session parameters:

- electrode impedance
- electrode shift
- channel dropout probability
- reference quality
- drift magnitude
- calibration duration
- class prior shift
- label noise
- marker delay

Acceptance:

- Domain-randomized sweeps produce interpretable failure modes.
- Subject profiles can be saved and reused.
- Reports distinguish subject variation from session variation.

#### Level 4: Source-Space Or Head-Model Simulation

Purpose:

- Add more plausible spatial structure where MNE support is available.

Implementation approach:

- Wrap MNE simulation tools rather than writing head modeling from scratch.
- Use simplified source components for MVP.
- Keep assumptions visible in reports.

Acceptance:

- Optional dependency only.
- If MNE simulation is unavailable, lower-level simulation still works.
- Generated output includes source/component metadata where possible.

#### Level 5: Closed-Loop User Model

Purpose:

- Test the entire feedback loop, not just classifier accuracy.

Features:

- User observes task feedback.
- User state changes with fatigue, reward, frustration, or learning.
- Decoder output affects task state.
- Task feedback affects subsequent simulated signal.
- Delay and jitter can destabilize performance.

Initial closed-loop target:

- 1D cursor control using motor imagery-like signals.

Acceptance:

- Static offline accuracy can diverge from closed-loop task success.
- Feedback delay and decoder uncertainty can affect simulated performance.
- Adaptation methods can be compared under controlled stressors.

### 7.2 Artifact Models

Initial artifacts:

- eye blink
- saccade
- muscle noise
- ECG contamination
- line noise
- electrode pop
- channel dropout
- saturation/clipping
- impedance-related noise
- movement burst
- marker jitter
- packet loss

Artifact outputs:

- additive signal contribution
- event labels
- sample masks
- channel masks
- artifact severity metadata

### 7.3 Drift Models

Initial drift models:

- slow amplitude drift
- spectral peak drift
- spatial covariance drift
- class-conditional drift
- electrode shift
- fatigue-linked SNR reduction
- attention-linked ERP amplitude reduction
- changing class priors

Drift should be configured independently from noise. This matters because systems can be robust to noise but brittle under covariance or class-conditional shift.

### 7.4 Timing Realism

Timing failure modes are core BCI failure modes.

Simulation and replay must support:

- marker delay
- marker jitter
- feedback delay
- packet batching
- dropped packets
- sampling-rate mismatch
- clock skew
- processing backlog
- scheduler jitter

Reports should show:

- sample timestamp to packet arrival latency
- node processing latency
- decoder decision latency
- feedback presentation latency
- p50, p95, p99 latency
- dropped packet counts
- backlog over time

### 7.5 Simulation Validation

Simulation credibility requires explicit validation and humility.

Validation should compare synthetic runs against public data where possible:

- PSD shape and bandpower distributions.
- ERP morphology and latency ranges.
- Spatial covariance structure.
- Cross-trial variability.
- Decoder performance under clean and stressed settings.
- Robustness curves under known perturbations.

Reports must label simulation level:

- `simulation_level: 0_plumbing`
- `simulation_level: 1_spectral`
- `simulation_level: 2_paradigm`
- `simulation_level: 3_subject_session`
- `simulation_level: 4_source_space`
- `simulation_level: 5_closed_loop`

Do not let a Level 1 generator masquerade as a validated human model.

## 8. Task And Environment Layer

A BCI task is not just a label source. It includes timing, goals, feedback, reward, success criteria, and adaptation hooks.

### 8.1 Task Contract

```python
class TaskEnvironment:
    def reset(self, context: RunContext) -> TaskStatePacket:
        ...

    def step(self, intent: IntentPacket, context: RunContext) -> tuple[TaskStatePacket, FeedbackPacket]:
        ...
```

Optional Gymnasium wrapper:

```python
env = task.to_gymnasium()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

### 8.2 Initial Tasks

#### Motor Imagery Classification

Purpose:

- Initial offline and synthetic MVP.

Features:

- left/right or left/right/rest classes.
- cue timing.
- trial windows.
- balanced and imbalanced class modes.

#### P300 Oddball Or Speller

Purpose:

- Event-related paradigm with target/non-target timing.

Features:

- target and non-target events.
- flash sequences.
- configurable inter-stimulus interval.
- simple row/column speller later.

#### 1D Cursor Control

Purpose:

- First closed-loop environment.

Features:

- decoder output moves cursor left/right.
- target acquisition.
- dwell time.
- feedback delay.
- reward.
- path efficiency.

#### SSVEP Target Selection

Purpose:

- Frequency-tagged paradigm and synthetic generator.

Features:

- target frequencies.
- harmonics.
- gaze/attention state.

### 8.3 Later Tasks

- 2D cursor control.
- ErrP-based correction.
- neurofeedback band-power regulation.
- workload detection.
- hybrid EEG + EMG control.
- invasive spike-count cursor control.

## 9. Evaluation And Reporting

The reporting layer should make practitioners trust the run. It should show more than accuracy.

### 9.1 Metrics

#### Decoder Metrics

- accuracy
- balanced accuracy
- AUC
- F1
- confusion matrix
- calibration error
- confidence quality
- cross-session degradation
- subject-transfer performance

#### BCI Task Metrics

- information transfer rate
- time to target
- target acquisition rate
- path efficiency
- false activation rate
- dwell time
- correction burden
- calibration time
- fatigue sensitivity

#### Systems Metrics

- end-to-end latency
- per-node latency
- jitter
- dropped samples
- marker delay
- synchronization error
- compute utilization
- streaming backlog
- warm-start time

#### Adaptation Metrics

- online regret
- recovery after drift
- update frequency
- stability under label noise
- catastrophic update warning
- labeled data required
- confidence-gated update quality

#### Simulation Metrics

- stressor settings
- latent subject state traces
- artifact severity
- generated SNR
- drift magnitude
- clean-vs-stressed score gap

### 9.2 Run Artifact Layout

Each run should write:

```text
runs/
  <run_id>/
    config.yml
    resolved_config.yml
    graph.json
    ontology_schema.json
    metrics.json
    telemetry.jsonl
    events.parquet
    predictions.parquet
    windows.parquet
    latency_trace.parquet
    adaptation.jsonl
    provenance.json
    model/
    report.html
    logs/
```

Use Parquet when available, but provide CSV fallback for early development if optional dependencies are missing.

### 9.3 Provenance

Provenance must include:

- git commit if available
- package version
- dependency versions
- platform
- Python version
- config hash
- random seeds
- source metadata
- optional dependency versions
- hardware metadata when live
- warnings and degraded-mode notes

### 9.4 Report Design

Report sections:

1. Run summary.
2. Experiment graph.
3. Source and data summary.
4. Decoder performance.
5. Closed-loop task performance.
6. Latency and jitter.
7. Drift and robustness.
8. Adaptation events.
9. Simulation ground truth when applicable.
10. Warnings and failure modes.
11. Reproducibility details.

HTML reports should be generated from the same JSON artifacts that power CLI summaries.

## 10. BCI StressBench

BCI StressBench should become the benchmark suite that complements MOABB. MOABB is strong for offline reproducible EEG benchmarking. StressBench should evaluate architecture-level, closed-loop, simulation-to-real robustness.

### 10.1 Stressor Presets

Initial presets:

- `clean`
- `low_snr`
- `high_blink`
- `muscle_noise`
- `channel_dropout`
- `electrode_shift`
- `session_drift`
- `subject_transfer`
- `delayed_feedback`
- `jittery_stream`
- `label_noise`
- `fatigue`
- `adaptive_user`

### 10.2 Scores

StressBench should report:

- offline score
- replay score
- closed-loop score
- robustness score
- latency score
- calibration efficiency
- adaptation stability
- failure threshold per stressor

### 10.3 Architecture Cards

Curated benchmark recipes:

- MI + CSP/LDA.
- MI + Riemannian MDM.
- MI + EEGNet.
- P300 + xDAWN/LDA.
- SSVEP + CCA.
- ErrP correction loop.
- Cursor control + Kalman-style decoder.
- Passive workload decoder.
- Hybrid EEG/EMG control.

These cards are important because they turn the workbench from generic plumbing into a credible BCI architecture lab.

## 11. Proposed Package Structure

```text
bciworkbench/
  __init__.py
  py.typed

  ontology/
    __init__.py
    schemas.py
    events.py
    channels.py
    packets.py
    timing.py
    validation.py

  graph/
    __init__.py
    node.py
    spec.py
    builder.py
    runtime.py
    scheduler.py
    clocks.py
    telemetry.py
    errors.py

  sources/
    __init__.py
    base.py
    synthetic.py
    mne.py
    moabb.py
    xdf.py
    lsl.py
    brainflow.py
    bids.py
    nwb.py

  sim/
    __init__.py
    config.py
    subjects.py
    artifacts.py
    rhythms.py
    paradigms.py
    motor_imagery.py
    p300.py
    ssvep.py
    drift.py
    timing.py
    domain_randomization.py
    closed_loop.py
    validation.py

  tasks/
    __init__.py
    base.py
    motor_imagery.py
    p300.py
    ssvep.py
    cursor.py
    neurofeedback.py
    gymnasium.py

  transforms/
    __init__.py
    base.py
    mne_adapters.py
    windowing.py
    filtering.py
    features.py
    channel_map.py

  decoders/
    __init__.py
    base.py
    sklearn.py
    pyriemann.py
    braindecode.py
    torch.py
    onnx.py
    kalman.py
    rl.py

  adaptation/
    __init__.py
    base.py
    supervised.py
    confidence_gated.py
    errp.py
    clda.py

  eval/
    __init__.py
    decoder_metrics.py
    task_metrics.py
    latency.py
    robustness.py
    reports.py
    compare.py

  standards/
    __init__.py
    bids.py
    nwb.py
    xdf.py
    provenance.py

  cli/
    __init__.py
    main.py
    run.py
    replay.py
    compare.py
    validate.py
    report.py

examples/
  mi_cursor_synthetic.yml
  mi_moabb_bnci2014_001.yml
  p300_synthetic.yml
  stressbench_mi.yml

tests/
  ontology/
  graph/
  sources/
  sim/
  tasks/
  decoders/
  eval/
  integration/
```

## 12. API Shape

### 12.1 Python API

```python
import bciworkbench as bci

experiment = bci.Experiment(
    name="mi_cursor_v1",
    source=bci.sources.SyntheticEEG(
        paradigm="motor_imagery",
        channels=32,
        sampling_rate=250,
        duration_s=300,
        seed=7,
    ),
    task=bci.tasks.Cursor1D(control="left_right"),
    pipeline=[
        bci.transforms.Bandpass(8, 30),
        bci.transforms.Window(length_s=1.0, stride_s=0.1),
        bci.transforms.CovarianceFeatures(),
        bci.decoders.SklearnDecoder("lda"),
    ],
    metrics=[
        "balanced_accuracy",
        "time_to_target",
        "latency_p95",
        "calibration_time",
    ],
)

run = experiment.run()
run.report().write_html()
```

### 12.2 Config API

```yaml
schema_version: 0.1
name: mi_cursor_baseline
paradigm: motor_imagery
mode: synthetic
random_seed: 7

source:
  type: synthetic_eeg
  paradigm: motor_imagery
  duration_s: 300
  channels: standard_10_20_32
  sampling_rate: 250
  subject:
    alpha_peak_hz: 10.2
    motor_imagery_vividness: 0.7
    fatigue_rate: 0.01
  session:
    snr_db: -6
    electrode_shift_mm: 4
    blink_rate_per_min: 12

pipeline:
  - type: bandpass
    fmin: 8
    fmax: 30
  - type: window
    length_s: 1.0
    stride_s: 0.1
  - type: covariance_features
  - type: sklearn_decoder
    estimator: lda

task:
  type: cursor_1d
  feedback_delay_ms: 80
  target_dwell_s: 0.5

adaptation:
  type: none

metrics:
  - balanced_accuracy
  - time_to_target
  - latency_p95
  - dropped_samples
```

### 12.3 CLI Shape

```text
bciworkbench validate examples/mi_cursor_synthetic.yml
bciworkbench run examples/mi_cursor_synthetic.yml
bciworkbench run examples/mi_cursor_synthetic.yml --source synthetic
bciworkbench run examples/mi_moabb_bnci2014_001.yml --source moabb:BNCI2014_001
bciworkbench replay data/session01.xdf --config examples/mi_cursor_synthetic.yml --speed 1.0
bciworkbench compare runs/* --report
bciworkbench stressbench examples/stressbench_mi.yml
```

## 13. Phased Implementation Plan

### Phase 0: Repository And Packaging Foundation

Goal:

Create a clean installable Python package with tests, linting, docs skeleton, and example configs.

Tasks:

- Add `pyproject.toml`.
- Add package skeleton under `bciworkbench/`.
- Add `tests/`.
- Add `examples/`.
- Add `README.md` with concise product positioning.
- Add `docs/` or MkDocs later, but do not start with heavy docs tooling.
- Configure `ruff`, `pytest`, and type checking.
- Add CLI entry point using Typer.
- Add CI-ready commands in README.

Definition of done:

- `pip install -e .` works.
- `pytest` runs.
- `bciworkbench --help` works.
- Empty package imports without optional dependencies.

### Phase 1: Ontology And Config Validation

Goal:

Implement the stable semantic contracts before broad functionality.

Tasks:

- Implement `ExperimentSpec`.
- Implement `RunContext`.
- Implement clock domain models.
- Implement `ChannelSchema`.
- Implement `Event`.
- Implement runtime packet dataclasses.
- Implement config loading from YAML.
- Implement JSON Schema export.
- Implement validation errors with useful messages.
- Add fixtures for motor imagery, P300, and simple cursor configs.

Definition of done:

- Example configs validate.
- Invalid configs fail with targeted errors.
- Packet dataclasses can serialize to artifact-friendly dictionaries.
- Ontology tests cover timing, channels, events, and packet shape.

### Phase 2: Deterministic Linear Runtime

Goal:

Run a simple graph in offline/synthetic mode and produce artifacts.

Tasks:

- Implement `Node`.
- Implement linear graph builder from config.
- Implement runtime context creation.
- Implement deterministic offline executor.
- Implement telemetry collection.
- Implement error handling.
- Implement artifact directory creation.
- Implement `Recorder` node.
- Implement `Evaluator` node placeholder.

Definition of done:

- A synthetic signal source can pass packets through windowing and a dummy decoder.
- Runtime records graph structure and telemetry.
- Failed nodes produce actionable errors.
- Fixed seed runs produce stable outputs.

### Phase 3: Basic Synthetic Source And Windowing

Goal:

Make the first runnable demo without external data.

Tasks:

- Implement Level 0 synthetic source.
- Implement Level 1 spectral EEG-like source.
- Implement basic motor imagery event generator.
- Implement basic P300 event generator.
- Implement windowing transform.
- Implement label assignment rules based on event timing.
- Implement simple bandpass transform using SciPy.
- Implement simple feature extraction:
  - bandpower
  - covariance

Definition of done:

- `examples/mi_cursor_synthetic.yml` runs end-to-end.
- Events, windows, and labels are inspectable.
- Synthetic data is deterministic under seed.
- Basic metrics are written.

### Phase 4: Decoder Adapters And Offline Baselines

Goal:

Support credible classical baselines before deep learning.

Tasks:

- Implement decoder base class.
- Implement `SklearnDecoder`.
- Implement sklearn LDA and logistic regression recipes.
- Implement optional `PyRiemannDecoder`.
- Implement train/predict split.
- Implement calibration window handling.
- Implement model save/load.
- Implement prediction artifact writing.
- Implement decoder metrics.

Definition of done:

- Synthetic MI baseline trains and predicts.
- P300 synthetic baseline trains and predicts.
- Metrics include accuracy, balanced accuracy, confusion matrix, and calibration duration.
- Decoder latency is measured.

### Phase 5: MNE And MOABB Offline Integration

Goal:

Move from synthetic-only to public dataset-backed runs.

Tasks:

- Implement `MNERawSource`.
- Implement conversion from MNE annotations to events.
- Implement MNE channel schema conversion.
- Implement `MOABBSource` for BNCI2014_001.
- Preserve subject/session/run metadata.
- Add example config for BNCI2014_001 motor imagery.
- Add tests guarded by optional dependency availability.

Definition of done:

- A MOABB BNCI2014_001 config can run through the same pipeline shape as synthetic MI.
- Source metadata appears in the report.
- No core import fails when MNE/MOABB are not installed.

### Phase 6: Reporting MVP

Goal:

Produce useful run reports before expanding integrations.

Tasks:

- Implement `metrics.json`.
- Implement `provenance.json`.
- Implement `events` and `predictions` artifact writers.
- Implement Jinja2 HTML report.
- Add CLI summary output.
- Add `compare` command for multiple runs.

Definition of done:

- Every completed run writes a readable report.
- Report includes graph, config, metrics, latency, source summary, warnings, and provenance.
- Compare command can compare synthetic runs under different stressor settings.

### Phase 7: Simulation Realism MVP

Goal:

Build the first real moat: task-aware, stressor-aware synthetic BCI data.

Tasks:

- Implement `SubjectProfile`.
- Implement `SessionProfile`.
- Implement motor imagery ERD/ERS model.
- Implement P300 ERP model.
- Implement artifact models:
  - blink
  - muscle noise
  - line noise
  - channel dropout
- Implement drift models:
  - amplitude drift
  - spectral drift
  - spatial covariance drift
- Implement timing stressors:
  - marker delay
  - marker jitter
  - feedback delay
- Implement `DomainRandomization`.
- Write stressor presets.

Definition of done:

- MI and P300 synthetic runs degrade plausibly under stressors.
- Stressor settings are logged as ground truth.
- Reports show robustness curves across at least three stressor presets.
- Simulation level is clearly labeled in artifacts and reports.

### Phase 8: Replay And Real-Time Timing

Goal:

Bridge offline recordings and real-time execution.

Tasks:

- Implement replay scheduler.
- Add speed modes: fastest, real-time, scaled, stepped.
- Implement packet arrival timestamp tracking.
- Implement queue and backlog telemetry.
- Implement `XDFReplaySource`.
- Implement latency trace artifact.
- Add stream health diagnostics.

Definition of done:

- Recorded streams can be replayed faster than real time and at real time.
- Latency and backlog are visible in reports.
- Marker timing is preserved separately from packet arrival time.

### Phase 9: LSL And BrainFlow Live Sources

Goal:

Support live data intake without changing downstream graph definitions.

Tasks:

- Implement `LSLSource`.
- Implement stream discovery CLI.
- Implement live source health telemetry.
- Implement `BrainFlowSource`.
- Add graceful shutdown behavior.
- Add partial run report writing.

Definition of done:

- A live LSL source emits `SignalPacket`s into the same graph runtime.
- A BrainFlow-supported board can be configured without importing BrainFlow in core.
- Live runs write latency, dropped-sample, and provenance artifacts.

### Phase 10: Closed-Loop Task Runtime

Goal:

Move from decoder benchmarking to whole-system BCI testing.

Tasks:

- Implement `TaskEnvironment`.
- Implement `Cursor1D`.
- Implement feedback packets.
- Implement closed-loop synthetic user model.
- Implement feedback delay modeling.
- Implement task metrics:
  - time to target
  - path efficiency
  - target acquisition rate
  - false activation rate
- Implement optional Gymnasium wrapper.

Definition of done:

- A synthetic MI cursor run can be evaluated by task success, not just decoder accuracy.
- Feedback delay changes task performance.
- Reports show the gap between decoder metrics and closed-loop metrics.

### Phase 11: Adaptation Interfaces

Goal:

Support controlled comparisons of static and adaptive BCI systems.

Tasks:

- Implement adaptation base class.
- Implement no-op adapter.
- Implement supervised batch recalibration.
- Implement confidence-gated update.
- Implement drift-triggered update hooks.
- Log `AdaptationPacket`s.
- Add adaptation stability metrics.

Definition of done:

- Static and adaptive decoders can be compared under session drift.
- Updates are logged with before/after metrics.
- Reports warn about unstable adaptation behavior.

### Phase 12: StressBench

Goal:

Create the benchmark suite that makes the project more than glue.

Tasks:

- Implement stressbench config format.
- Implement stressor preset registry.
- Implement run matrix generation.
- Implement robustness scoring.
- Implement architecture cards:
  - MI bandpower + LDA
  - MI covariance + pyRiemann MDM
  - P300 bandpass/window + LDA
- Implement stressbench report.

Definition of done:

- `bciworkbench stressbench examples/stressbench_mi.yml` runs a matrix of synthetic stressors.
- Report identifies which stressors break an architecture.
- Scores include accuracy, latency, robustness, and calibration efficiency.

### Phase 13: Documentation And Examples

Goal:

Make the library understandable and usable.

Tasks:

- Write README quickstart.
- Write ontology guide.
- Write source adapter guide.
- Write simulation realism guide.
- Write report interpretation guide.
- Add example notebooks only after CLI examples are stable.
- Document non-goals and limitations.

Definition of done:

- A new user can run a synthetic example in under 10 minutes.
- Documentation clearly states what simulation levels mean.
- Integration docs explain optional extras and conversion semantics.

## 14. Testing Strategy

### Unit Tests

- Ontology validation.
- Event timing.
- Channel schema transformations.
- Packet serialization.
- Node setup/process/teardown.
- Graph validation.
- Synthetic source determinism.
- Artifact model output shapes.
- Metrics correctness.

### Golden Tests

- Fixed-seed synthetic run produces stable metrics within tolerances.
- Example config produces expected artifact set.
- HTML report contains required sections.

### Integration Tests

Guarded optional tests:

- MNE raw conversion.
- MOABB source loading.
- pyRiemann decoder.
- XDF replay.
- LSL source using a local test stream.
- BrainFlow synthetic board if supported.

### Simulation Validation Tests

- PSD has expected broad shape.
- P300 component appears in target-locked average.
- MI ERD/ERS effect changes bandpower in expected windows.
- Increasing noise decreases decoder performance on average.
- Increasing drift worsens cross-session performance on average.

### Real-Time Tests

- Replay scheduler respects speed settings.
- Queue backlog telemetry records overload.
- Dropped packet simulation is visible in report.
- Graceful shutdown writes partial artifacts.

## 15. Main Risks And Mitigations

### Risk: Ontology Becomes Too Broad

Mitigation:

- Start with MI, P300, and cursor control.
- Keep extension metadata namespaced.
- Add schema only when an actual adapter or task needs it.
- Reject vague abstractions that cannot be tested.

### Risk: Simulation Looks Real But Is Misleading

Mitigation:

- Label simulation level in every artifact.
- Emit ground truth stressor settings.
- Add validation reports against public data.
- Use MNE source-space simulation where appropriate instead of pretending a simple generator is biophysical.
- Document assumptions clearly.

### Risk: Project Becomes Thin Wrappers

Mitigation:

- Prioritize graph runtime, ontology, reports, and StressBench.
- Use integrations to feed the architecture layer, not as the product itself.
- Every adapter must improve simulation-to-real parity or reproducibility.

### Risk: Real-Time Runtime Gets Complex Too Early

Mitigation:

- Build deterministic offline mode first.
- Add replay mode before live mode.
- Use the same packet and telemetry contracts across all modes.
- Keep live scheduler conservative and observable.

### Risk: Optional Dependencies Become Fragile

Mitigation:

- Use extras.
- Lazy import optional dependencies.
- Add guarded tests.
- Keep core import dependency-light.
- Pin known-good integration versions in development docs.

### Risk: Reports Become Cosmetic

Mitigation:

- Reports must be generated from structured artifacts.
- CLI summaries and HTML reports should use the same data.
- Reports must surface warnings, timing, and failures, not just nice charts.

## 16. Immediate Implementation Order

The first coding pass should implement only the foundation needed to prove the shape:

1. Add packaging, tests, and CLI skeleton.
2. Implement ontology models and validation.
3. Implement deterministic linear runtime.
4. Implement Level 0 and Level 1 synthetic source.
5. Implement windowing, simple bandpower features, and sklearn LDA decoder.
6. Implement basic metrics and run artifacts.
7. Add one runnable synthetic motor imagery example.
8. Generate the first basic report.

This first milestone should answer:

> Can one config define a BCI-like experiment, execute deterministically, and produce trustworthy artifacts?

Only after that works should the project add MOABB, MNE, XDF/LSL, richer simulation, and closed-loop tasks.

## 17. Milestone 1 Deliverable

Milestone 1 should produce:

```text
pip install -e .
bciworkbench validate examples/mi_synthetic.yml
bciworkbench run examples/mi_synthetic.yml
bciworkbench report runs/<run_id>
pytest
```

Expected output:

- A validated experiment config.
- A deterministic synthetic motor imagery run.
- A trained baseline decoder.
- Metrics written to JSON.
- Events, windows, and predictions written to tabular artifacts.
- A simple HTML report.
- Tests covering ontology, runtime, synthetic source, decoder, and metrics.

## 18. Success Criteria For The Project

The project is on the right track when:

- A user can swap `source: synthetic` for `source: moabb` without rewriting the pipeline.
- A replayed recording and a live stream share the same downstream graph.
- Reports make timing and drift visible, not hidden.
- Synthetic stressors produce meaningful robustness curves.
- The ontology prevents common BCI mistakes such as mixing cue, target, response, intent, and feedback.
- Integrations feel like native use of MNE, MOABB, LSL, BrainFlow, pyRiemann, Braindecode, and Gymnasium.
- Benchmark recipes make architecture comparisons reproducible.

The v0.1 wedge should be narrow and strong:

> EEG motor imagery and P300, synthetic plus offline public data, deterministic reports, and the first robustness stressors.

That is enough to prove the project without pretending to be the whole BCI ecosystem.
