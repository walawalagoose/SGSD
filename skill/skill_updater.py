from __future__ import annotations

import json
from typing import Any

from .llm_client import build_chat_client
from .merge_math_skills import (
    dedupe_exact_items,
    merge_collection,
    normalize_common_mistake,
    normalize_general_skill,
    reassign_ids,
    serialize_common_mistakes_for_merge,
    serialize_general_skills_for_merge,
)
from .prompts import render_prompt
from .utils import default_planning_pattern, infer_failure_mode, split_reasoning_steps, squeeze_whitespace, summarize_step_action


SUCCESS_PROMPT = "success_skill_from_memory_prompt.txt"
FAILURE_PROMPT = "failure_skill_from_memory_prompt.txt"
GENERAL_MERGE_PROMPT = "merge_general_skills_prompt.txt"
MISTAKE_MERGE_PROMPT = "merge_common_mistakes_prompt.txt"

MAX_ITEMS_PER_TRAJECTORY = 3
EXTRACTION_BATCH_SIZE = 8
MERGE_BATCH_SIZE = 8
MERGE_GROUP_SIZE = 32
MERGE_STAGNATION_PATIENCE = 3


class CallbackChatClient:
    def __init__(self, batch_complete_fn) -> None:
        self._batch_complete_fn = batch_complete_fn

    def complete(self, prompt: str) -> str:
        return self.batch_complete([prompt])[0]

    def batch_complete(self, prompts: list[str]) -> list[str]:
        return self._batch_complete_fn(prompts)


class MathSkillUpdater:
    def __init__(
        self,
        max_new_skills_per_update: int = 5,
        max_completion_tokens: int = 2048,
        llm_backend: str = "azure",
        model: str = "o3",
        local_device_map: str = "auto",
        local_dtype: str = "bfloat16",
        local_attn_implementation: str = "flash_attention_2",
        local_temperature: float = 0.0,
        local_top_p: float = 1.0,
        local_tensor_parallel_size: int = 1,
        local_gpu_memory_utilization: float = 0.9,
        local_max_model_len: int | None = None,
        dynamic_skill_capacity: int = 30,
    ) -> None:
        self.max_new_skills_per_update = int(max_new_skills_per_update)
        self.dynamic_skill_capacity = int(dynamic_skill_capacity)
        self.max_completion_tokens = int(max_completion_tokens)
        self.local_temperature = float(local_temperature)
        self.local_top_p = float(local_top_p)
        if self.max_new_skills_per_update < 0:
            raise ValueError("max_new_skills_per_update must be non-negative.")
        if self.dynamic_skill_capacity <= 0:
            raise ValueError("dynamic_skill_capacity must be positive.")

        normalized_backend = llm_backend.strip().lower().replace("-", "_")
        effective_model = "o3" if normalized_backend in {"azure", "azure_openai"} and model is None else model
        self._uses_shared_main_model = normalized_backend in {
            "local",
            "local_hf",
            "transformers",
            "vllm",
            "local_vllm",
        } and model is None
        self._shared_client = None
        self.client = None
        if not self._uses_shared_main_model:
            self.client = build_chat_client(
                backend=llm_backend,
                model=effective_model,
                max_completion_tokens=max_completion_tokens,
                local_device_map=local_device_map,
                local_dtype=local_dtype,
                local_attn_implementation=local_attn_implementation,
                local_temperature=local_temperature,
                local_top_p=local_top_p,
                local_tensor_parallel_size=local_tensor_parallel_size,
                local_gpu_memory_utilization=local_gpu_memory_utilization,
                local_max_model_len=local_max_model_len,
            )

    def analyze_trajectories(
        self,
        *,
        successful_records: list[dict[str, Any]],
        failed_records: list[dict[str, Any]],
        current_skills: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        current_dyn_general = self._get_dynamic_general_skills(current_skills)
        current_dyn_mistakes = self._get_dynamic_common_mistakes(current_skills)

        new_general_candidates = self._extract_general_candidates(successful_records)
        merged_new_general = self._merge_general_skills(new_general_candidates)
        updated_dyn_general = self._merge_with_existing_dynamic_items(
            existing_dynamic_items=current_dyn_general,
            new_items=merged_new_general,
            kind="general",
        )

        new_mistake_candidates = self._extract_common_mistake_candidates(failed_records)
        merged_new_mistakes = self._merge_common_mistakes(new_mistake_candidates)
        updated_dyn_mistakes = self._merge_with_existing_dynamic_items(
            existing_dynamic_items=current_dyn_mistakes,
            new_items=merged_new_mistakes,
            kind="mistake",
        )

        return {
            "general_skills": updated_dyn_general,
            "common_mistakes": updated_dyn_mistakes,
        }

    def uses_shared_main_model(self) -> bool:
        return self._uses_shared_main_model

    def set_shared_client(self, client) -> None:
        self._shared_client = client

    def clear_shared_client(self) -> None:
        self._shared_client = None

    def _extract_general_candidates(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._extract_candidates_from_records(
            records=records,
            prompt_name=SUCCESS_PROMPT,
            output_key="general_skills",
            normalize_item=normalize_general_skill,
            dedupe_keys=("title", "principle", "when_to_apply"),
            outcome="Success",
        )

    def _extract_common_mistake_candidates(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._extract_candidates_from_records(
            records=records,
            prompt_name=FAILURE_PROMPT,
            output_key="common_mistakes",
            normalize_item=normalize_common_mistake,
            dedupe_keys=("description", "why_it_happens", "how_to_avoid"),
            outcome="Failure",
        )

    def _extract_candidates_from_records(
        self,
        *,
        records: list[dict[str, Any]],
        prompt_name: str,
        output_key: str,
        normalize_item,
        dedupe_keys: tuple[str, ...],
        outcome: str,
    ) -> list[dict[str, Any]]:
        if not records:
            return []

        prompts = [
            render_prompt(
                prompt_name,
                memory_json=json.dumps(
                    self._record_to_online_memory(record, outcome=outcome, index=index),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            for index, record in enumerate(records, start=1)
        ]

        candidates: list[dict[str, Any]] = []
        client = self._get_client()
        for start in range(0, len(prompts), EXTRACTION_BATCH_SIZE):
            batch_prompts = prompts[start : start + EXTRACTION_BATCH_SIZE]
            responses = client.batch_complete(batch_prompts)
            if len(responses) != len(batch_prompts):
                raise RuntimeError(
                    f"Expected {len(batch_prompts)} candidate responses, got {len(responses)}."
                )

            for response in responses:
                try:
                    parsed = self._parse_response_to_json_object(response)
                except (ValueError, json.JSONDecodeError):
                    continue

                normalized_items = [
                    normalized
                    for normalized in (normalize_item(item) for item in parsed.get(output_key, [])[:MAX_ITEMS_PER_TRAJECTORY])
                    if normalized is not None
                ]
                candidates.extend(normalized_items)

        return dedupe_exact_items(candidates, dedupe_keys)

    def _merge_general_skills(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._merge_items(
            items=items,
            prompt_name=GENERAL_MERGE_PROMPT,
            output_key="general_skills",
            serialize_items=serialize_general_skills_for_merge,
            normalize_item=normalize_general_skill,
            dedupe_keys=("title", "principle", "when_to_apply"),
        )

    def _merge_common_mistakes(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._merge_items(
            items=items,
            prompt_name=MISTAKE_MERGE_PROMPT,
            output_key="common_mistakes",
            serialize_items=serialize_common_mistakes_for_merge,
            normalize_item=normalize_common_mistake,
            dedupe_keys=("description", "why_it_happens", "how_to_avoid"),
        )

    def _merge_items(
        self,
        *,
        items: list[dict[str, Any]],
        prompt_name: str,
        output_key: str,
        serialize_items,
        normalize_item,
        dedupe_keys: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        normalized_items = [normalized for normalized in (normalize_item(item) for item in items) if normalized is not None]
        normalized_items = dedupe_exact_items(normalized_items, dedupe_keys)
        if not normalized_items:
            return []

        client = self._get_client()
        merged_items, _ = merge_collection(
            client=client,
            items=normalized_items,
            prompt_name=prompt_name,
            output_key=output_key,
            serialize_items=serialize_items,
            normalize_item=normalize_item,
            batch_size=MERGE_BATCH_SIZE,
            group_size=MERGE_GROUP_SIZE,
            stagnation_patience=MERGE_STAGNATION_PATIENCE,
        )
        return dedupe_exact_items(merged_items, dedupe_keys)

    def _merge_with_existing_dynamic_items(
        self,
        *,
        existing_dynamic_items: list[dict[str, Any]],
        new_items: list[dict[str, Any]],
        kind: str,
    ) -> list[dict[str, Any]]:
        if not new_items:
            return existing_dynamic_items

        combined_items = list(existing_dynamic_items) + list(new_items)
        if kind == "general":
            merged_items = self._merge_general_skills(combined_items)
            id_field = "skill_id"
        elif kind == "mistake":
            merged_items = self._merge_common_mistakes(combined_items)
            id_field = "mistake_id"
        else:
            raise ValueError(f"Unsupported dynamic item kind: {kind!r}")

        if not merged_items:
            return existing_dynamic_items

        pruned_items = self._prune_dynamic_items(
            merged_items,
            previous_count=len(existing_dynamic_items),
        )
        return reassign_ids(pruned_items, id_field, "dyn")

    def _prune_dynamic_items(
        self,
        items: list[dict[str, Any]],
        *,
        previous_count: int,
    ) -> list[dict[str, Any]]:
        max_allowed_after_growth = previous_count + self.max_new_skills_per_update
        pruned_items = list(items)
        if len(pruned_items) > max_allowed_after_growth:
            pruned_items = pruned_items[:max_allowed_after_growth]
        if len(pruned_items) > self.dynamic_skill_capacity:
            pruned_items = pruned_items[: self.dynamic_skill_capacity]
        return pruned_items

    def _get_dynamic_general_skills(self, current_skills: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            normalized
            for normalized in (
                normalize_general_skill(skill)
                for skill in current_skills.get("general_skills", [])
                if str(skill.get("skill_id", "")).startswith("dyn_")
            )
            if normalized is not None
        ]

    def _get_dynamic_common_mistakes(self, current_skills: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            normalized
            for normalized in (
                normalize_common_mistake(mistake)
                for mistake in current_skills.get("common_mistakes", [])
                if str(mistake.get("mistake_id", "")).startswith("dyn_")
            )
            if normalized is not None
        ]

    def _record_to_online_memory(
        self,
        record: dict[str, Any],
        *,
        outcome: str,
        index: int,
    ) -> dict[str, Any]:
        completion = str(record.get("completion", "") or "")
        task = str(record.get("task", "") or "")
        predicted_answer = record.get("predicted_answer")
        ground_truth = str(record.get("ground_truth", "") or "")

        step_texts = split_reasoning_steps(completion, max_steps=6)
        compact_trajectory = [
            {
                "step": step_index,
                "action": summarize_step_action(step_text),
                "observation": step_text,
            }
            for step_index, step_text in enumerate(step_texts, start=1)
        ]

        memory = {
            "memory_id": f"online_{outcome.lower()}_{index:06d}",
            "goal": task,
            "task": task,
            "outcome": outcome,
            "data_source": str(record.get("data_source", "math") or "math"),
            "planning_pattern": default_planning_pattern(completion),
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truth,
            "reward": float(record.get("reward", 0.0) or 0.0),
            "completion_excerpt": self._truncate_text(completion, max_chars=1600 if outcome == "Success" else 2200),
            "trajectory": compact_trajectory,
        }

        if outcome == "Success":
            memory["feedback"] = "Reached a correct final answer consistent with the ground truth."
            memory["refined_trajectory"] = compact_trajectory
        else:
            failure_mode, failure_feedback = infer_failure_mode(completion, predicted_answer, ground_truth)
            memory["feedback"] = failure_feedback
            memory["mistakes_to_avoid"] = [failure_mode]

        return memory

    @staticmethod
    def _truncate_text(text: Any, max_chars: int) -> str:
        normalized = squeeze_whitespace(str(text or ""))
        if not normalized or len(normalized) <= max_chars:
            return normalized
        head = max_chars // 2
        tail = max_chars - head - len(" ...[snip]... ")
        tail = max(tail, 0)
        return f"{normalized[:head].rstrip()} ...[snip]... {normalized[-tail:].lstrip()}"

    @staticmethod
    def _parse_response_to_json_object(response: str) -> dict[str, Any]:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            raise ValueError("Model response does not contain a JSON object.")
        return json.loads(response[json_start:json_end])

    def _get_client(self):
        if self._shared_client is not None:
            return self._shared_client
        if self.client is not None:
            return self.client
        raise RuntimeError(
            "MathSkillUpdater requires a shared main-model client, but none is attached. "
            "This should only happen if the trainer forgot to bind the SGSD subagent client."
        )
