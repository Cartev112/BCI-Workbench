from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bciworkbench.graph.context import RunContext
from bciworkbench.ontology.packets import FeedbackPacket, IntentPacket, TaskStatePacket
from bciworkbench.sources.base import OptionalDependencyError
from bciworkbench.tasks.user import DelayedIntentUserModel, intent_direction


@dataclass(frozen=True)
class Cursor1DConfig:
    target_position: float = 1.0
    target_radius: float = 0.2
    target_dwell_steps: int = 1
    step_size: float = 1.0
    control_interval_s: float = 0.5
    feedback_delay_ms: float = 0.0
    confidence_threshold: float = 0.0
    reset_on_target_change: bool = True

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "Cursor1DConfig":
        return cls(
            target_position=float(params.get("target_position", 1.0)),
            target_radius=float(params.get("target_radius", 0.2)),
            target_dwell_steps=int(params.get("target_dwell_steps", params.get("target_dwell", 1))),
            step_size=float(params.get("step_size", 1.0)),
            control_interval_s=float(params.get("control_interval_s", 0.5)),
            feedback_delay_ms=float(params.get("feedback_delay_ms", 0.0)),
            confidence_threshold=float(params.get("confidence_threshold", 0.0)),
            reset_on_target_change=bool(params.get("reset_on_target_change", True)),
        )

    def __post_init__(self) -> None:
        if self.target_position <= 0:
            raise ValueError("target_position must be positive")
        if self.target_radius < 0:
            raise ValueError("target_radius must be non-negative")
        if self.target_dwell_steps <= 0:
            raise ValueError("target_dwell_steps must be positive")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.control_interval_s <= 0:
            raise ValueError("control_interval_s must be positive")
        if self.feedback_delay_ms < 0:
            raise ValueError("feedback_delay_ms must be non-negative")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")


@dataclass(frozen=True)
class CursorTaskResult:
    states: list[TaskStatePacket]
    feedback: list[FeedbackPacket]
    metrics: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)


class Cursor1DEnvironment:
    """A minimal 1D cursor task driven by decoded left/right intents."""

    def __init__(self, config: Cursor1DConfig | None = None) -> None:
        self.config = config or Cursor1DConfig()
        self.user = DelayedIntentUserModel(
            feedback_delay_ms=self.config.feedback_delay_ms,
            control_interval_s=self.config.control_interval_s,
            confidence_threshold=self.config.confidence_threshold,
        )
        self.position = 0.0
        self.target_label: str | None = None
        self.target_direction = 0
        self.target_started_step = 0
        self.dwell_steps = 0
        self.step_index = 0
        self.path_length = 0.0
        self.rows: list[dict[str, Any]] = []

    def reset(self, context: RunContext | None = None) -> TaskStatePacket:
        del context
        self.user.reset()
        self.position = 0.0
        self.target_label = None
        self.target_direction = 0
        self.target_started_step = 0
        self.dwell_steps = 0
        self.step_index = 0
        self.path_length = 0.0
        self.rows = []
        return self._state(success=False, reward=0.0, done=False)

    def step(
        self,
        intent: IntentPacket,
        context: RunContext | None = None,
    ) -> tuple[TaskStatePacket, FeedbackPacket]:
        del context
        label = intent.label or intent.intent
        next_direction = intent_direction(label)
        if next_direction and (self.target_label != label or self.target_direction != next_direction):
            self._set_target(label, next_direction)

        applied_direction, applied_confidence, applied_intent = self.user.choose_action(intent)
        previous = self.position
        delta = applied_direction * self.config.step_size * applied_confidence
        self.position = float(np.clip(self.position + delta, -self.config.target_position, self.config.target_position))
        self.path_length += abs(self.position - previous)

        target_position = self.target_direction * self.config.target_position
        distance = abs(target_position - self.position) if self.target_direction else None
        in_target = bool(distance is not None and distance <= self.config.target_radius)
        self.dwell_steps = self.dwell_steps + 1 if in_target else 0
        success = self.dwell_steps >= self.config.target_dwell_steps
        false_activation = bool(applied_direction and self.target_direction and applied_direction != self.target_direction)
        reward = 1.0 if success else (-0.1 if false_activation else -0.01)
        elapsed_s = (self.step_index - self.target_started_step + 1) * self.config.control_interval_s

        state = self._state(
            success=success,
            reward=reward,
            done=success,
            label=label,
            intent=intent.intent,
            applied_intent=applied_intent,
            applied_direction=applied_direction,
            false_activation=false_activation,
            elapsed_s=elapsed_s,
        )
        feedback = FeedbackPacket(
            action=applied_intent,
            rendered_at=(self.step_index + 1) * self.config.control_interval_s + self.config.feedback_delay_ms / 1000.0,
            clock_domain="sim_clock",
            reward=reward,
            delay_ms=self.config.feedback_delay_ms,
            task_state=state,
            metadata={
                "intent_id": intent.intent_id,
                "decoder_intent": intent.intent,
                "decoder_confidence": intent.confidence,
                "applied_direction": applied_direction,
                "target_direction": self.target_direction,
            },
        )
        self.rows.append(
            {
                "step_index": self.step_index,
                "intent_id": intent.intent_id,
                "window_id": intent.window_id,
                "label": label,
                "decoder_intent": intent.intent,
                "applied_intent": applied_intent,
                "confidence": intent.confidence,
                "position": self.position,
                "target_position": target_position,
                "success": success,
                "reward": reward,
                "false_activation": false_activation,
                "elapsed_s": elapsed_s,
                "feedback_delay_ms": self.config.feedback_delay_ms,
            }
        )
        self.step_index += 1
        return state, feedback

    def metrics(self, states: list[TaskStatePacket], feedback: list[FeedbackPacket]) -> dict[str, Any]:
        del feedback
        task_states = states[1:] if states and states[0].metadata.get("reset") else states
        if not task_states:
            return {
                "closed_loop_n_steps": 0,
                "simulation_level": "5_closed_loop",
                "target_acquisition_rate": None,
                "mean_time_to_target_s": None,
                "path_efficiency": None,
                "false_activation_rate": None,
                "mean_task_reward": None,
                "feedback_delay_ms": self.config.feedback_delay_ms,
            }
        successes = [state for state in task_states if state.success]
        false_activations = [row for row in self.rows if row["false_activation"]]
        time_to_target = [state.metadata["elapsed_s"] for state in successes if "elapsed_s" in state.metadata]
        direct_success_distance = len(successes) * self.config.target_position
        path_efficiency = None
        if self.path_length > 0 and successes:
            path_efficiency = min(1.0, direct_success_distance / self.path_length)
        rewards = [state.reward for state in task_states if state.reward is not None]
        return {
            "closed_loop_n_steps": len(task_states),
            "simulation_level": "5_closed_loop",
            "target_success_count": len(successes),
            "target_acquisition_rate": len(successes) / len(task_states),
            "mean_time_to_target_s": sum(time_to_target) / len(time_to_target) if time_to_target else None,
            "path_efficiency": path_efficiency,
            "false_activation_rate": len(false_activations) / len(task_states),
            "mean_task_reward": sum(rewards) / len(rewards) if rewards else None,
            "feedback_delay_ms": self.config.feedback_delay_ms,
        }

    def to_gymnasium(self):
        try:
            import gymnasium as gym
            from gymnasium import spaces
        except Exception as exc:
            raise OptionalDependencyError(
                'Cursor1D gymnasium wrapper requires gymnasium. Install with: pip install "gymnasium"'
            ) from exc

        env = self

        class _CursorGymEnv(gym.Env):
            observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            action_space = spaces.Discrete(3)

            def reset(self, seed=None, options=None):
                del seed, options
                state = env.reset()
                return _observation(state), {"state": state.to_dict()}

            def step(self, action):
                intent = _intent_from_action(int(action), env.step_index)
                state, feedback = env.step(intent)
                return _observation(state), float(feedback.reward or 0.0), state.done, False, {"state": state.to_dict()}

        return _CursorGymEnv()

    def _set_target(self, label: str, direction: int) -> None:
        self.target_label = label
        self.target_direction = direction
        self.target_started_step = self.step_index
        self.dwell_steps = 0
        if self.config.reset_on_target_change:
            self.position = 0.0

    def _state(
        self,
        success: bool,
        reward: float,
        done: bool,
        label: str | None = None,
        intent: str | None = None,
        applied_intent: str | None = None,
        applied_direction: int = 0,
        false_activation: bool = False,
        elapsed_s: float = 0.0,
    ) -> TaskStatePacket:
        target_position = self.target_direction * self.config.target_position
        return TaskStatePacket(
            task_id="cursor_1d",
            state={
                "position": self.position,
                "target_position": target_position,
                "dwell_steps": self.dwell_steps,
                "step_index": self.step_index,
            },
            observation={"cursor": self.position, "target": target_position},
            target=label or self.target_label,
            reward=reward,
            done=done,
            success=success,
            metadata={
                "reset": label is None and intent is None and self.step_index == 0,
                "decoder_intent": intent,
                "applied_intent": applied_intent,
                "applied_direction": applied_direction,
                "target_direction": self.target_direction,
                "false_activation": false_activation,
                "elapsed_s": elapsed_s,
                "feedback_delay_ms": self.config.feedback_delay_ms,
            },
        )


def run_cursor_task(predictions: list[IntentPacket], params: dict[str, Any]) -> CursorTaskResult:
    env = Cursor1DEnvironment(Cursor1DConfig.from_params(params))
    states = [env.reset()]
    feedback: list[FeedbackPacket] = []
    for prediction in predictions:
        state, packet = env.step(prediction)
        states.append(state)
        feedback.append(packet)
    return CursorTaskResult(states=states, feedback=feedback, metrics=env.metrics(states, feedback), rows=env.rows)


def _observation(state: TaskStatePacket) -> np.ndarray:
    return np.asarray([state.observation.get("cursor", 0.0), state.observation.get("target", 0.0)], dtype=np.float32)


def _intent_from_action(action: int, index: int) -> IntentPacket:
    labels = {0: "left", 1: "noop", 2: "right"}
    label = labels.get(action, "noop")
    return IntentPacket(
        intent_id=f"gym-action-{index:04d}",
        intent=label,
        confidence=1.0 if label != "noop" else 0.0,
        posterior={label: 1.0},
        latency_ms=0.0,
        window_id=f"gym-window-{index:04d}",
        decoder_id="gym_action",
        label=label,
    )
