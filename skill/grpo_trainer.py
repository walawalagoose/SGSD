from __future__ import annotations

import copy
from typing import Any

from trl import GRPOTrainer

from .training import SkillRuntime


class SkillAwareGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        skill_runtime: SkillRuntime | None = None,
        **kwargs,
    ) -> None:
        self.skill_runtime = skill_runtime
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        prepared_inputs = copy.deepcopy(inputs)
        if self.skill_runtime is not None:
            for example in prepared_inputs:
                example["prompt"] = self.skill_runtime.inject_prompt(example["prompt"])

        return super()._generate_and_score_completions(prepared_inputs)
