from __future__ import annotations

import json
import os
import random
from typing import Any

from .utils import stable_int_hash


class MathSkillsOnlyMemory:
    def __init__(
        self,
        skills_json_path: str,
        retrieval_mode: str = "embedding",
        embedding_model_path: str | None = None,
        random_seed: int = 42,
    ) -> None:
        if retrieval_mode not in {"embedding", "random"}:
            raise ValueError(f"retrieval_mode must be 'embedding' or 'random', got {retrieval_mode!r}")
        if not os.path.exists(skills_json_path):
            raise FileNotFoundError(f"Skills file not found: {skills_json_path}")

        self.skills_json_path = skills_json_path
        self.retrieval_mode = retrieval_mode
        self.embedding_model_path = embedding_model_path or "Qwen/Qwen3-Embedding-0.6B"
        self.random_seed = int(random_seed)

        self.skills: dict[str, Any] = {}
        self._embedding_model = None
        self._embeddings_cache: dict[str, Any] | None = None
        self.reload_skills(skills_json_path)

    def reload_skills(self, path: str | None = None) -> None:
        target = path or self.skills_json_path
        with open(target, "r", encoding="utf-8") as f:
            self.skills = json.load(f)
        self.skills_json_path = target
        self._embeddings_cache = None

    def _get_embedding_model(self):
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for embedding retrieval. "
                    "Install it with `pip install sentence-transformers`."
                ) from exc
            self._embedding_model = SentenceTransformer(self.embedding_model_path)
        return self._embedding_model

    @staticmethod
    def _skill_to_text(skill: dict[str, Any]) -> str:
        return MathSkillsOnlyMemory._item_to_text(skill, ("title", "principle", "when_to_apply"))

    @staticmethod
    def _mistake_to_text(mistake: dict[str, Any]) -> str:
        return MathSkillsOnlyMemory._item_to_text(mistake, ("description", "why_it_happens", "how_to_avoid"))

    @staticmethod
    def _item_to_text(item: dict[str, Any], fields: tuple[str, ...]) -> str:
        parts = []
        for field in fields:
            value = str(item.get(field, "") or "").strip()
            if value:
                parts.append(value)
        return ". ".join(parts)

    def _compute_embeddings(self) -> dict[str, Any]:
        if self._embeddings_cache is not None:
            return self._embeddings_cache

        general_skills = self.skills.get("general_skills", [])
        common_mistakes = self.skills.get("common_mistakes", [])
        model = self._get_embedding_model()

        self._embeddings_cache = {
            "general": {
                "items": general_skills,
                "embeddings": self._encode_items(model, [self._skill_to_text(skill) for skill in general_skills]),
            },
            "mistakes": {
                "items": common_mistakes,
                "embeddings": self._encode_items(
                    model,
                    [self._mistake_to_text(mistake) for mistake in common_mistakes],
                ),
            },
        }
        return self._embeddings_cache

    @staticmethod
    def _encode_items(model, texts: list[str]):
        if not texts:
            return None
        return model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def retrieve(self, task_description: str, top_k: int = 6, **_: Any) -> dict[str, Any]:
        if self.retrieval_mode == "random":
            general_skills = self._random_retrieve_general(task_description, top_k)
            common_mistakes = self._random_retrieve_mistakes(task_description, top_k)
            return {
                "general_skills": general_skills,
                "task_specific_skills": [],
                "mistakes_to_avoid": common_mistakes,
                "task_type": "general_math",
                "task_specific_examples": [],
                "retrieval_mode": "random",
            }

        general_skills = self._embedding_retrieve(task_description, top_k, category="general")
        common_mistakes = self._embedding_retrieve(task_description, top_k, category="mistakes")
        return {
            "general_skills": general_skills,
            "task_specific_skills": [],
            "mistakes_to_avoid": common_mistakes,
            "task_type": "general_math",
            "task_specific_examples": [],
            "retrieval_mode": "embedding",
        }

    def _random_retrieve_general(self, task_description: str, top_k: int) -> list[dict[str, Any]]:
        all_general = list(self.skills.get("general_skills", []))
        dynamic_skills = [skill for skill in all_general if str(skill.get("skill_id", "")).startswith("dyn_")]
        static_skills = [skill for skill in all_general if not str(skill.get("skill_id", "")).startswith("dyn_")]
        prioritized_dynamic = dynamic_skills[:top_k]
        remaining = max(0, top_k - len(prioritized_dynamic))

        rng = random.Random(self.random_seed + stable_int_hash(task_description))
        if len(static_skills) <= remaining:
            sampled_static = static_skills
        else:
            sampled_static = rng.sample(static_skills, remaining)
        return prioritized_dynamic + sampled_static

    def _random_retrieve_mistakes(self, task_description: str, top_k: int) -> list[dict[str, Any]]:
        all_mistakes = list(self.skills.get("common_mistakes", []))
        dynamic_mistakes = [
            mistake for mistake in all_mistakes if str(mistake.get("mistake_id", "")).startswith("dyn_")
        ]
        static_mistakes = [
            mistake for mistake in all_mistakes if not str(mistake.get("mistake_id", "")).startswith("dyn_")
        ]
        prioritized_dynamic = dynamic_mistakes[:top_k]
        remaining = max(0, top_k - len(prioritized_dynamic))

        rng = random.Random(self.random_seed + stable_int_hash(f"mistakes::{task_description}"))
        if len(static_mistakes) <= remaining:
            sampled_static = static_mistakes
        else:
            sampled_static = rng.sample(static_mistakes, remaining)
        return prioritized_dynamic + sampled_static

    def _embedding_retrieve(self, task_description: str, top_k: int, *, category: str) -> list[dict[str, Any]]:
        ranked = self._embedding_retrieve_with_scores(task_description, top_k, category=category)
        return [item for item, _ in ranked]

    def _embedding_retrieve_with_scores(
        self,
        task_description: str,
        top_k: int,
        *,
        category: str,
    ) -> list[tuple[dict[str, Any], float]]:
        import numpy as np

        cache = self._compute_embeddings()
        category_cache = cache[category]
        if not category_cache["items"] or category_cache["embeddings"] is None:
            return []

        model = self._get_embedding_model()
        query_emb = model.encode(
            [task_description],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        sims = category_cache["embeddings"] @ query_emb
        idx = np.argsort(sims)[::-1][:top_k]
        return [(category_cache["items"][int(i)], float(sims[int(i)])) for i in idx]

    def retrieve_individual_items(self, task_description: str, top_k: int = 6) -> list[dict[str, Any]]:
        if self.retrieval_mode == "random":
            general_candidates = [
                {
                    "kind": "general",
                    "item": skill,
                    "score": None,
                    "rank": rank,
                }
                for rank, skill in enumerate(self._random_retrieve_general(task_description, top_k), start=1)
            ]
            mistake_candidates = [
                {
                    "kind": "mistake",
                    "item": mistake,
                    "score": None,
                    "rank": rank,
                }
                for rank, mistake in enumerate(self._random_retrieve_mistakes(task_description, top_k), start=1)
            ]
            return self._interleave_candidates(general_candidates, mistake_candidates)

        general_candidates = [
            {
                "kind": "general",
                "item": skill,
                "score": score,
                "rank": rank,
            }
            for rank, (skill, score) in enumerate(
                self._embedding_retrieve_with_scores(task_description, top_k, category="general"),
                start=1,
            )
        ]
        mistake_candidates = [
            {
                "kind": "mistake",
                "item": mistake,
                "score": score,
                "rank": rank,
            }
            for rank, (mistake, score) in enumerate(
                self._embedding_retrieve_with_scores(task_description, top_k, category="mistakes"),
                start=1,
            )
        ]
        return self._interleave_candidates(general_candidates, mistake_candidates)

    def retrieve_ranked_items_by_kind(self, task_description: str, top_k: int = 6) -> dict[str, list[dict[str, Any]]]:
        if self.retrieval_mode == "random":
            general_candidates = [
                {
                    "kind": "general",
                    "item": skill,
                    "score": None,
                    "rank": rank,
                }
                for rank, skill in enumerate(self._random_retrieve_general(task_description, top_k), start=1)
            ]
            mistake_candidates = [
                {
                    "kind": "mistake",
                    "item": mistake,
                    "score": None,
                    "rank": rank,
                }
                for rank, mistake in enumerate(self._random_retrieve_mistakes(task_description, top_k), start=1)
            ]
        else:
            general_candidates = [
                {
                    "kind": "general",
                    "item": skill,
                    "score": score,
                    "rank": rank,
                }
                for rank, (skill, score) in enumerate(
                    self._embedding_retrieve_with_scores(task_description, top_k, category="general"),
                    start=1,
                )
            ]
            mistake_candidates = [
                {
                    "kind": "mistake",
                    "item": mistake,
                    "score": score,
                    "rank": rank,
                }
                for rank, (mistake, score) in enumerate(
                    self._embedding_retrieve_with_scores(task_description, top_k, category="mistakes"),
                    start=1,
                )
            ]
        return {"general": general_candidates, "mistake": mistake_candidates}

    @staticmethod
    def _interleave_candidates(
        general_candidates: list[dict[str, Any]],
        mistake_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        combined = []
        max_len = max(len(general_candidates), len(mistake_candidates))
        for index in range(max_len):
            if index < len(general_candidates):
                combined.append(general_candidates[index])
            if index < len(mistake_candidates):
                combined.append(mistake_candidates[index])
        return combined

    def format_for_prompt(self, retrieved_memories: dict[str, Any]) -> str:
        sections = []

        general_skills = retrieved_memories.get("general_skills", [])
        if general_skills:
            lines = ["### General Principles"]
            for skill in general_skills:
                title = skill.get("title", "")
                principle = skill.get("principle", "")
                when = skill.get("when_to_apply", "")
                lines.append(f"- **{title}**: {principle}")
                if when:
                    lines.append(f"  _Apply when: {when}_")
            sections.append("\n".join(lines))

        mistakes = retrieved_memories.get("mistakes_to_avoid", [])
        if mistakes:
            lines = ["### Mistakes to Avoid"]
            for mistake in mistakes:
                desc = mistake.get("description", "")
                fix = mistake.get("how_to_avoid", "")
                if desc:
                    lines.append(f"- **Don't**: {desc}")
                if fix:
                    lines.append(f"  **Instead**: {fix}")
            sections.append("\n".join(lines))

        if not sections:
            return ""

        return (
            "You may use the following retrieved math-reasoning skills as soft guidance. "
            "Solve the current problem independently and do not quote the skills verbatim.\n\n"
            + "\n\n".join(sections)
        )

    def format_individual_item_for_prompt(self, candidate: dict[str, Any]) -> str:
        kind = str(candidate.get("kind", "") or "").strip().lower()
        item = candidate.get("item") or {}

        if kind == "general":
            title = str(item.get("title", "") or "").strip()
            principle = str(item.get("principle", "") or "").strip()
            when = str(item.get("when_to_apply", "") or "").strip()
            if not any((title, principle, when)):
                return ""
            lines = ["### General Principle"]
            lines.append(f"- **{title}**: {principle}" if title else f"- {principle}")
            if when:
                lines.append(f"  _Apply when: {when}_")
        elif kind == "mistake":
            desc = str(item.get("description", "") or "").strip()
            fix = str(item.get("how_to_avoid", "") or "").strip()
            why = str(item.get("why_it_happens", "") or "").strip()
            if not any((desc, fix, why)):
                return ""
            lines = ["### Mistake to Avoid"]
            if desc:
                lines.append(f"- **Don't**: {desc}")
            if fix:
                lines.append(f"  **Instead**: {fix}")
            if why:
                lines.append(f"  _Why this matters: {why}_")
        else:
            return ""

        return (
            "You may use the following retrieved math-reasoning guidance as soft guidance. "
            "Solve the current problem independently and do not quote it verbatim.\n\n"
            + "\n".join(lines)
        )

    def format_paired_items_for_prompt(
        self,
        *,
        general_candidate: dict[str, Any] | None,
        mistake_candidate: dict[str, Any] | None,
    ) -> str:
        sections = []
        if general_candidate is not None:
            general_block = self.format_individual_item_for_prompt(general_candidate)
            if general_block:
                parts = general_block.split("\n\n", maxsplit=1)
                sections.append(parts[-1])
        if mistake_candidate is not None:
            mistake_block = self.format_individual_item_for_prompt(mistake_candidate)
            if mistake_block:
                parts = mistake_block.split("\n\n", maxsplit=1)
                sections.append(parts[-1])

        if not sections:
            return ""

        return (
            "You may use the following retrieved math-reasoning guidance as soft guidance. "
            "Solve the current problem independently and do not quote it verbatim.\n\n"
            + "\n\n".join(sections)
        )

    def add_skills(self, new_skills: list[dict[str, Any]], category: str = "general") -> int:
        del category
        added = 0
        existing_ids = self._get_all_skill_ids()
        for skill in new_skills:
            skill_id = skill.get("skill_id")
            if not skill_id or skill_id in existing_ids:
                continue
            self.skills.setdefault("general_skills", []).append(skill)
            existing_ids.add(skill_id)
            added += 1
        if added:
            self._embeddings_cache = None
        return added

    def get_dynamic_skills(self) -> list[dict[str, Any]]:
        return [
            dict(skill)
            for skill in self.skills.get("general_skills", [])
            if str(skill.get("skill_id", "")).startswith("dyn_")
        ]

    def replace_dynamic_skills(self, dynamic_skills: list[dict[str, Any]]) -> None:
        static_skills = [
            skill
            for skill in self.skills.get("general_skills", [])
            if not str(skill.get("skill_id", "")).startswith("dyn_")
        ]
        self.skills["general_skills"] = static_skills + list(dynamic_skills)
        self._embeddings_cache = None

    def get_dynamic_mistakes(self) -> list[dict[str, Any]]:
        return [
            dict(mistake)
            for mistake in self.skills.get("common_mistakes", [])
            if str(mistake.get("mistake_id", "")).startswith("dyn_")
        ]

    def replace_dynamic_mistakes(self, dynamic_mistakes: list[dict[str, Any]]) -> None:
        static_mistakes = [
            mistake
            for mistake in self.skills.get("common_mistakes", [])
            if not str(mistake.get("mistake_id", "")).startswith("dyn_")
        ]
        self.skills["common_mistakes"] = static_mistakes + list(dynamic_mistakes)
        self._embeddings_cache = None

    def save_skills(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.skills, f, indent=2, ensure_ascii=False)

    def _get_all_skill_ids(self) -> set[str]:
        ids = set()
        for skill in self.skills.get("general_skills", []):
            skill_id = skill.get("skill_id")
            if skill_id:
                ids.add(skill_id)
        return ids
