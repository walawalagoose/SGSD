from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset


@dataclass(frozen=True)
class DatasetModeSpec:
    mode: str
    question_field: str
    reward_solution_field: str
    teacher_solution_field: str | None
    teacher_solution_kind: str
    requires_boxed_teacher_solution: bool


MODE_SPECS: dict[str, DatasetModeSpec] = {
    "openthought": DatasetModeSpec(
        mode="openthought",
        question_field="problem",
        reward_solution_field="Answer",
        teacher_solution_field="solution",
        teacher_solution_kind="reasoning_trace",
        requires_boxed_teacher_solution=True,
    ),
    "dapo-math": DatasetModeSpec(
        mode="dapo-math",
        question_field="prompt",
        reward_solution_field="reward_model.ground_truth",
        teacher_solution_field=None,
        teacher_solution_kind="final_answer_only",
        requires_boxed_teacher_solution=True,
    ),
}


DATASET_NAME_TO_MODE: dict[str, str] = {
    "siyanzhao/openthoughts_math_30k_opsd": "openthought",
    "openthoughts_math_30k_opsd": "openthought",
    "bytedtsinghua-sia/dapo-math-17k": "dapo-math",
    "dapo-math-17k": "dapo-math",
    "dapo-math-17k-processed": "dapo-math",
    "dapo-math": "dapo-math",
    "dapo-math-12k": "dapo-math",
}

DATASET_MODE_TO_SUBSET: dict[str, str | None] = {
    "openthought": None,
    "dapo-math": None,
}


def _normalize_name(dataset_name: str) -> str:
    return dataset_name.strip().rstrip("/").lower()


def _name_candidates(dataset_name: str) -> list[str]:
    normalized = _normalize_name(dataset_name)
    return [normalized, Path(normalized).name]


def resolve_dataset_mode(dataset_name: str) -> str:
    for candidate in _name_candidates(dataset_name):
        mode = DATASET_NAME_TO_MODE.get(candidate)
        if mode is not None:
            return mode

    known = ", ".join(sorted(DATASET_NAME_TO_MODE))
    raise ValueError(
        f"Unrecognized dataset_name '{dataset_name}'. "
        f"Please register it in DATASET_NAME_TO_MODE. Known mappings: {known}"
    )


def resolve_dataset_subset(dataset_mode: str) -> str | None:
    if dataset_mode not in DATASET_MODE_TO_SUBSET:
        known = ", ".join(sorted(DATASET_MODE_TO_SUBSET))
        raise ValueError(f"Unsupported dataset mode '{dataset_mode}'. Known modes: {known}")
    return DATASET_MODE_TO_SUBSET[dataset_mode]


def load_dataset_with_mode(dataset_name: str):
    dataset_mode = resolve_dataset_mode(dataset_name)
    dataset_subset_name = resolve_dataset_subset(dataset_mode)
    if dataset_subset_name is None:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, name=dataset_subset_name)
    return dataset_mode, dataset_subset_name, dataset


def _require_text(value: Any, *, dataset_mode: str, role: str, field_name: str) -> str:
    if value is None:
        raise KeyError(f"Dataset mode '{dataset_mode}' requires '{field_name}' for {role}.")
    if not isinstance(value, str):
        raise TypeError(
            f"Dataset mode '{dataset_mode}' expects '{field_name}' to be str for {role}, "
            f"but got {type(value).__name__}."
        )
    return value


def _require_text_field(example: dict[str, Any], field_name: str, dataset_mode: str, role: str) -> str:
    return _require_text(
        example.get(field_name),
        dataset_mode=dataset_mode,
        role=role,
        field_name=field_name,
    )


def _extract_user_prompt(messages: Any, dataset_mode: str) -> str:
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        raise TypeError(
            f"Dataset mode '{dataset_mode}' expects 'prompt' to be str or a message list, "
            f"but got {type(messages).__name__}."
        )

    text_messages: list[str] = []
    user_message: str | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            text_messages.append(content)
            if message.get("role") == "user":
                user_message = content
    if user_message is not None:
        return user_message
    if text_messages:
        return text_messages[-1]
    raise ValueError(f"Dataset mode '{dataset_mode}' could not find text content in 'prompt'.")


def extract_question(example: dict[str, Any], dataset_mode: str) -> str:
    if dataset_mode == "openthought":
        return _require_text_field(example, "problem", dataset_mode, role="question")
    if dataset_mode == "dapo-math":
        if "prompt" not in example:
            raise KeyError("Dataset mode 'dapo-math' requires 'prompt' for question.")
        return _extract_user_prompt(example["prompt"], dataset_mode)
    raise ValueError(f"Unknown dataset mode: {dataset_mode}.")


def extract_reward_solution(example: dict[str, Any], dataset_mode: str) -> str:
    if dataset_mode == "openthought":
        return _require_text_field(example, "Answer", dataset_mode, role="reward ground truth")
    if dataset_mode == "dapo-math":
        reward_model = example.get("reward_model")
        if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
            return _require_text(
                reward_model["ground_truth"],
                dataset_mode=dataset_mode,
                role="reward ground truth",
                field_name="reward_model.ground_truth",
            )
        return _require_text_field(example, "solution", dataset_mode, role="reward ground truth")
    raise ValueError(f"Unknown dataset mode: {dataset_mode}.")


def _contains_boxed(text: str) -> bool:
    return r"\boxed{" in text


def _boxed_from_answer(answer: str) -> str:
    answer = answer.strip()
    if _contains_boxed(answer):
        return answer
    return rf"\boxed{{{answer}}}"


def extract_teacher_solution(example: dict[str, Any], dataset_mode: str) -> str:
    spec = MODE_SPECS.get(dataset_mode)
    if spec is None:
        raise ValueError(f"Unknown dataset mode: {dataset_mode}.")

    if spec.teacher_solution_field is not None:
        teacher_solution = _require_text_field(
            example,
            spec.teacher_solution_field,
            dataset_mode,
            role="teacher solution",
        )
    else:
        teacher_solution = f"The answer is {_boxed_from_answer(extract_reward_solution(example, dataset_mode))}."

    if spec.requires_boxed_teacher_solution and not _contains_boxed(teacher_solution):
        raise ValueError(
            f"Dataset mode '{dataset_mode}' requires teacher solution text to contain '\\\\boxed{{}}'."
        )
    return teacher_solution


def make_grpo_format_prompt(tokenizer, dataset_mode: str):
    if dataset_mode not in MODE_SPECS:
        raise ValueError(f"Unknown dataset mode: {dataset_mode}.")

    def format_prompt(example: dict[str, Any]) -> dict[str, str]:
        question = extract_question(example, dataset_mode)
        ground_truth = extract_reward_solution(example, dataset_mode)
        messages = [
            {
                "role": "user",
                "content": (
                    f"Problem: {question}\n"
                    "Please reason step by step, and put your final answer within \\boxed{}."
                ),
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "ground_truth": ground_truth}

    return format_prompt


def make_opsd_format_example(dataset_mode: str):
    if dataset_mode not in MODE_SPECS:
        raise ValueError(f"Unknown dataset mode: {dataset_mode}.")
    spec = MODE_SPECS[dataset_mode]

    def format_example(example: dict[str, Any]) -> dict[str, str]:
        return {
            "problem": extract_question(example, dataset_mode),
            "solution": extract_teacher_solution(example, dataset_mode),
            "reward_solution": extract_reward_solution(example, dataset_mode),
            "solution_kind": spec.teacher_solution_kind,
        }

    return format_example
