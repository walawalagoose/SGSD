from __future__ import annotations

import torch

from .training import SkillRuntime


class SGSDDataCollator:
    """
    Data collator for SGSD.

    Student prompt: problem only.
    Teacher contexts: one rank-aligned skill and mistake pair per teacher.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        skill_runtime: SkillRuntime | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.skill_runtime = skill_runtime
        self.tokenizer.padding_side = "right"

    def __call__(self, features):
        student_prompts = []
        problems = []
        ground_truths = []
        teacher_reference_solutions = []
        sgsd_teacher_problem_contexts = []
        sgsd_teacher_weights = []

        for feature in features:
            problem = feature["problem"]
            solution = feature["solution"]
            ground_truth = feature.get("reward_solution", feature.get("ground_truth", ""))

            student_user_message = (
                f"Problem: {problem}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
            student_messages = [{"role": "user", "content": student_user_message}]
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            student_prompts.append(student_prompt)

            problems.append(problem)
            ground_truths.append(str(ground_truth))
            teacher_reference_solutions.append(str(solution))
            if self.skill_runtime is not None:
                sgsd_teacher_specs = self.skill_runtime.build_sgsd_teacher_specs(problem)
            else:
                sgsd_teacher_specs = []
            sgsd_teacher_problem_contexts.append([spec["problem_context"] for spec in sgsd_teacher_specs])
            sgsd_teacher_weights.append([float(spec["weight"]) for spec in sgsd_teacher_specs])

        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]
        max_student_prompt_len = max(student_prompt_lengths)
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        return {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
            "problems": problems,
            "ground_truths": ground_truths,
            "teacher_reference_solutions": teacher_reference_solutions,
            "sgsd_teacher_problem_contexts": sgsd_teacher_problem_contexts,
            "sgsd_teacher_weights": sgsd_teacher_weights,
        }
