from __future__ import annotations

import copy
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from data.dataset_adapter import MODE_SPECS, extract_question, extract_reward_solution

from .skill_memory import MathSkillsOnlyMemory
from .skill_updater import MathSkillUpdater
from .utils import extract_problem_text, inject_skill_block


@dataclass
class SkillRetrievalArguments:
    use_skills: bool = field(
        default=False,
        metadata={"help": "Enable SkillRL-style math skill retrieval for training prompts."},
    )
    skills_json_path: str | None = field(
        default=None,
        metadata={"help": "Path to the initialized math skill library JSON."},
    )
    skills_retrieval_mode: str = field(
        default="embedding",
        metadata={"help": "Skill retrieval mode. Supported: `embedding`, `random`."},
    )
    skills_embedding_model_path: str = field(
        default="Qwen/Qwen3-Embedding-0.6B",
        metadata={"help": "Embedding model path used when `skills_retrieval_mode=embedding`."},
    )
    skills_top_k: int = field(
        default=8,
        metadata={"help": "Number of general skills and common mistakes retrieved for each prompt."},
    )
    skills_random_seed: int = field(
        default=42,
        metadata={"help": "Base seed for deterministic random-skill retrieval."},
    )


@dataclass
class SkillArguments(SkillRetrievalArguments):
    skills_enable_dynamic_update: bool = field(
        default=False,
        metadata={
            "help": "Enable online dynamic updates for both general skills and common mistakes during training."
        },
    )
    skills_update_backend: str = field(
        default="vllm",
        metadata={"help": "Backend used for online skill writing. Supported: `azure`, `local`, `vllm`."},
    )
    skills_update_model: str | None = field(
        default=None,
        metadata={"help": "Model name or path used by the online skill writer. Defaults to the training model."},
    )
    skills_update_local_device_map: str = field(
        default="auto",
        metadata={"help": "Device map used when `skills_update_backend=local`."},
    )
    skills_update_local_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Torch dtype used when `skills_update_backend=local`."},
    )
    skills_update_local_attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation used when `skills_update_backend=local`."},
    )
    skills_update_local_temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature used when `skills_update_backend=local`."},
    )
    skills_update_local_top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p used when `skills_update_backend=local`."},
    )
    skills_update_local_tensor_parallel_size: int = field(
        default=4,
        metadata={"help": "Tensor parallel size used when `skills_update_backend=vllm`."},
    )
    skills_update_local_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={"help": "GPU memory utilization used when `skills_update_backend=vllm`."},
    )
    skills_update_local_max_model_len: int | None = field(
        default=None,
        metadata={"help": "Optional max model length used when `skills_update_backend=vllm`."},
    )
    skills_update_threshold: float = field(
        default=0.8,
        metadata={"help": "Update when the observed success rate falls below this threshold."},
    )
    skills_update_frequency: int = field(
        default=25,
        metadata={"help": "Check whether to update the skill bank every N optimizer steps."},
    )
    skills_max_new_skills: int = field(
        default=5,
        metadata={
            "help": (
                "Maximum net increase in dynamic (`dyn_###`) items allowed per update after "
                "the online merge pipeline finishes, applied separately to `general_skills` and `common_mistakes`."
            )
        },
    )
    skills_max_failures_to_analyze: int = field(
        default=-1,
        metadata={
            "help": (
                "Maximum number of successful trajectories and failed trajectories to send to the skill updater per "
                "cycle, applied separately to each group. "
                "Set to -1 to use all available records from the current update window."
            )
        },
    )
    skills_dynamic_capacity: int = field(
        default=30,
        metadata={
            "help": (
                "Maximum number of dynamic (`dyn_###`) items retained after each online update for "
                "each collection (`general_skills` and `common_mistakes`)."
            )
        },
    )
    skills_save_path: str | None = field(
        default=None,
        metadata={"help": "Optional fixed path for the latest dynamically updated skill bank."},
    )


def make_grpo_raw_prompt_example(dataset_mode: str):
    if dataset_mode not in MODE_SPECS:
        raise ValueError(f"Unknown dataset mode: {dataset_mode}.")
    def format_example(example: dict[str, Any]) -> dict[str, Any]:
        question = extract_question(example, dataset_mode)
        ground_truth = extract_reward_solution(example, dataset_mode)
        data_source = str(example.get("data_source", dataset_mode))
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"Problem: {question}\n"
                        "Please reason step by step, and put your final answer within \\boxed{}."
                    ),
                }
            ],
            "ground_truth": ground_truth,
            "data_source": data_source,
        }

    return format_example


class SkillRuntime:
    def __init__(self, args: SkillRetrievalArguments) -> None:
        if not args.skills_json_path:
            raise ValueError("`skills_json_path` must be provided when `use_skills=True`.")

        self.args = args
        self.memory = MathSkillsOnlyMemory(
            skills_json_path=args.skills_json_path,
            retrieval_mode=args.skills_retrieval_mode,
            embedding_model_path=args.skills_embedding_model_path,
            random_seed=args.skills_random_seed,
        )
        self.updater = (
            MathSkillUpdater(
                max_new_skills_per_update=args.skills_max_new_skills,
                llm_backend=args.skills_update_backend,
                model=args.skills_update_model,
                local_device_map=args.skills_update_local_device_map,
                local_dtype=args.skills_update_local_dtype,
                local_attn_implementation=args.skills_update_local_attn_implementation,
                local_temperature=args.skills_update_local_temperature,
                local_top_p=args.skills_update_local_top_p,
                local_tensor_parallel_size=args.skills_update_local_tensor_parallel_size,
                local_gpu_memory_utilization=args.skills_update_local_gpu_memory_utilization,
                local_max_model_len=args.skills_update_local_max_model_len,
                dynamic_skill_capacity=args.skills_dynamic_capacity,
            )
            if getattr(args, "skills_enable_dynamic_update", False)
            else None
        )
        self._last_update_step = -1
        self._online_update_enabled = False

    def inject_prompt(self, prompt: Any) -> Any:
        task_description = extract_problem_text(prompt)
        retrieved = self.memory.retrieve(task_description=task_description, top_k=self.args.skills_top_k)
        skill_block = self.memory.format_for_prompt(retrieved)
        return inject_skill_block(copy.deepcopy(prompt), skill_block)

    def inject_problem(self, problem: str) -> str:
        retrieved = self.memory.retrieve(task_description=problem, top_k=self.args.skills_top_k)
        skill_block = self.memory.format_for_prompt(retrieved)
        if not skill_block:
            return problem
        return f"{skill_block}\n\nProblem: {problem}"

    def build_sgsd_teacher_specs(self, problem: str) -> list[dict[str, Any]]:
        ranked_candidates = self.memory.retrieve_ranked_items_by_kind(
            task_description=problem,
            top_k=self.args.skills_top_k,
        )
        general_candidates = ranked_candidates.get("general", [])
        mistake_candidates = ranked_candidates.get("mistake", [])
        pair_count = min(len(general_candidates), len(mistake_candidates))
        if pair_count <= 0:
            return []

        paired_candidates = []
        for rank, (general_candidate, mistake_candidate) in enumerate(
            zip(general_candidates[:pair_count], mistake_candidates[:pair_count], strict=True),
            start=1,
        ):
            guidance_block = self.memory.format_for_prompt(
                {
                    "general_skills": [general_candidate["item"]],
                    "mistakes_to_avoid": [mistake_candidate["item"]],
                }
            )
            if not guidance_block:
                continue
            scores = [
                float(candidate["score"])
                for candidate in (general_candidate, mistake_candidate)
                if candidate.get("score") is not None
            ]
            paired_candidates.append(
                {
                    "problem_context": f"{guidance_block}\n\nProblem: {problem}",
                    "rank": rank,
                    "score": (sum(scores) / len(scores)) if scores else None,
                }
            )
        if not paired_candidates:
            return []

        weights = self._normalize_sgsd_candidate_weights(paired_candidates)
        for candidate, weight in zip(paired_candidates, weights, strict=True):
            candidate["weight"] = float(weight)
        return paired_candidates

    @staticmethod
    def _normalize_sgsd_candidate_weights(candidates: list[dict[str, Any]]) -> list[float]:
        if not candidates:
            return []

        scores = [candidate.get("score") for candidate in candidates]
        if all(score is None for score in scores):
            uniform = 1.0 / len(candidates)
            return [uniform] * len(candidates)

        numeric_scores = [float(score) if score is not None else 0.0 for score in scores]
        max_score = max(numeric_scores)
        exp_scores = [math.exp(score - max_score) for score in numeric_scores]
        denom = sum(exp_scores)
        if denom <= 0:
            uniform = 1.0 / len(candidates)
            return [uniform] * len(candidates)
        return [score / denom for score in exp_scores]

    @contextmanager
    def online_update_scope(self, *, enabled: bool, shared_client=None):
        previous_enabled = self._online_update_enabled
        self._online_update_enabled = bool(enabled)
        if self.updater is not None and shared_client is not None:
            self.updater.set_shared_client(shared_client)
        try:
            yield
        finally:
            if self.updater is not None and shared_client is not None:
                self.updater.clear_shared_client()
            self._online_update_enabled = previous_enabled

    def maybe_update_from_records(
        self,
        records: list[dict[str, Any]],
        *,
        step: int,
        output_dir: str,
        accelerator: Any | None = None,
    ) -> None:
        if (
            self.updater is None
            or not records
            or not self._online_update_enabled
            or step <= 0
            or step % self.args.skills_update_frequency != 0
            or step == self._last_update_step
        ):
            return

        if accelerator is not None and hasattr(accelerator, "sync_gradients") and not accelerator.sync_gradients:
            return

        all_records = self._gather_records(records, accelerator)
        if not all_records:
            return

        success_rate = sum(1 for record in all_records if float(record.get("reward", 0.0)) > 0) / len(all_records)
        save_path = self._resolve_save_path(output_dir)

        if accelerator is None or accelerator.is_main_process:
            if success_rate < self.args.skills_update_threshold:
                successful_records = [
                    record
                    for record in all_records
                    if float(record.get("reward", 0.0)) > 0
                ]
                failed_records = [
                    record
                    for record in all_records
                    if float(record.get("reward", 0.0)) <= 0
                ]
                if self.args.skills_max_failures_to_analyze >= 0:
                    successful_records = successful_records[: self.args.skills_max_failures_to_analyze]
                    failed_records = failed_records[: self.args.skills_max_failures_to_analyze]
                current_dyn_skills = self.memory.get_dynamic_skills()
                current_dyn_mistakes = self.memory.get_dynamic_mistakes()
                updated_dynamic = self.updater.analyze_trajectories(
                    successful_records=successful_records,
                    failed_records=failed_records,
                    current_skills=self.memory.skills,
                )
                updated_dyn_skills = updated_dynamic["general_skills"]
                updated_dyn_mistakes = updated_dynamic["common_mistakes"]
                if updated_dyn_skills != current_dyn_skills or updated_dyn_mistakes != current_dyn_mistakes:
                    self.memory.replace_dynamic_skills(updated_dyn_skills)
                    self.memory.replace_dynamic_mistakes(updated_dyn_mistakes)
                    self.memory.save_skills(save_path)
                    snapshot_path = os.path.join(
                        os.path.dirname(save_path),
                        f"updated_skills_step{step}.json",
                    )
                    if snapshot_path != save_path:
                        self.memory.save_skills(snapshot_path)

        if accelerator is not None:
            accelerator.wait_for_everyone()

        if os.path.exists(save_path):
            self.memory.reload_skills(save_path)
        self._last_update_step = step

    def _gather_records(self, records: list[dict[str, Any]], accelerator: Any | None) -> list[dict[str, Any]]:
        if accelerator is None:
            return records

        from accelerate.utils import gather_object

        gathered = gather_object(records)
        flattened = []
        for item in gathered:
            if isinstance(item, list):
                flattened.extend(item)
            elif item is not None:
                flattened.append(item)
        return flattened

    def _resolve_save_path(self, output_dir: str) -> str:
        if self.args.skills_save_path:
            return self.args.skills_save_path
        return os.path.join(output_dir, "skill", "updated_skills_latest.json")


def build_skill_runtime(args: SkillRetrievalArguments) -> SkillRuntime | None:
    if not args.use_skills:
        return None
    return SkillRuntime(args)
