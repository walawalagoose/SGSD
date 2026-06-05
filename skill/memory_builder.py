from __future__ import annotations

import uuid
from typing import Any

from .utils import (
    default_planning_pattern,
    infer_failure_mode,
    split_reasoning_steps,
    squeeze_whitespace,
    summarize_step_action,
)


def build_memory_record(
    *,
    question: str,
    completion_text: str,
    reward_result: dict[str, Any],
    dataset_mode: str,
    data_source: str,
    origin_env_id: str,
) -> dict[str, Any]:
    correct = bool(reward_result.get("correct", False))
    predicted_answer = reward_result.get("pred_answer")
    ground_truth = str(reward_result.get("ground_truth", "") or "")
    refined_trajectory = _build_refined_trajectory(completion_text) if correct else None

    return {
        "memory_id": f"mem_math_{uuid.uuid4().hex[:8]}",
        "contextual_description": _build_contextual_description(
            question=question,
            completion_text=completion_text,
            correct=correct,
            data_source=data_source,
            predicted_answer=predicted_answer,
            ground_truth=ground_truth,
        ),
        "tags": {
            "environment": "MathReasoningAgent",
            "outcome": "Success" if correct else "Failure",
            "data_source": data_source,
        },
        "content": {
            "task_meta": {
                "original_goal": question,
                "data_source": data_source,
                "dataset_mode": dataset_mode,
                "ground_truth": ground_truth,
            },
            "refined_trajectory": refined_trajectory,
            "strategic_guidelines": {
                "strategic_guidelines": _build_strategic_guidelines(
                    completion_text=completion_text,
                    correct=correct,
                    predicted_answer=predicted_answer,
                    ground_truth=ground_truth,
                )
            },
            "raw_attempt": {
                "completion": completion_text,
                "predicted_answer": predicted_answer,
                "ground_truth": ground_truth,
                "reward": float(reward_result.get("reward", 0.0)),
                "feedback": reward_result.get("feedback"),
            },
        },
        "origin_env_id": origin_env_id,
    }


def _build_contextual_description(
    *,
    question: str,
    completion_text: str,
    correct: bool,
    data_source: str,
    predicted_answer: str | None,
    ground_truth: str,
) -> str:
    short_goal = squeeze_whitespace(question)
    if len(short_goal) > 220:
        short_goal = short_goal[:217] + "..."

    if correct:
        return (
            f"Math reasoning task from {data_source} solved successfully. The agent parsed the problem, "
            f"constructed a multistep derivation, and reached a verifiable final answer for: {short_goal}"
        )

    return (
        f"Math reasoning task from {data_source} ended in failure. The agent attempted a multistep derivation for: "
        f"{short_goal}, but produced the incorrect or unverifiable final answer {predicted_answer!r} "
        f"instead of the ground truth {ground_truth!r}."
    )


def _build_refined_trajectory(completion_text: str) -> list[dict[str, Any]]:
    steps = split_reasoning_steps(completion_text, max_steps=6)
    trajectory = []
    for idx, step in enumerate(steps, start=1):
        trajectory.append(
            {
                "step_index": idx,
                "action": summarize_step_action(step),
                "critical_observation": step[:280],
                "reasoning": "Advance the derivation while preserving consistency with the problem constraints.",
            }
        )
    return trajectory


def _build_strategic_guidelines(
    *,
    completion_text: str,
    correct: bool,
    predicted_answer: str | None,
    ground_truth: str,
) -> dict[str, Any]:
    if correct:
        return {
            "planning_pattern": default_planning_pattern(completion_text),
            "mistakes_to_avoid": [],
        }

    trigger_condition, bad_action = infer_failure_mode(completion_text, predicted_answer, ground_truth)
    return {
        "planning_pattern": None,
        "mistakes_to_avoid": [
            {
                "trigger_condition": trigger_condition,
                "bad_action": bad_action,
            }
        ],
    }
