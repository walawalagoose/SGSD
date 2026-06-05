from __future__ import annotations

import hashlib
import re
from typing import Any


def squeeze_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_thinking_tags(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def stable_int_hash(text: str) -> int:
    digest = hashlib.md5((text or "").encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def extract_problem_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return _strip_problem_prefix(content)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                return _strip_problem_prefix("\n".join(text_parts))
        return ""
    if isinstance(prompt, str):
        return _strip_problem_prefix(prompt)
    return squeeze_whitespace(str(prompt))


def _strip_problem_prefix(text: str) -> str:
    text = text or ""
    if "Problem:" in text:
        text = text.split("Problem:", 1)[1]
    return text.strip()


def inject_skill_block(prompt: Any, skill_block: str) -> Any:
    if not skill_block:
        return prompt

    if isinstance(prompt, list):
        updated = []
        replaced = False
        for idx, message in enumerate(prompt):
            cloned = dict(message) if isinstance(message, dict) else message
            if (
                not replaced
                and isinstance(cloned, dict)
                and cloned.get("role") == "user"
                and idx == len(prompt) - 1
            ):
                content = cloned.get("content", "")
                if isinstance(content, str):
                    cloned["content"] = f"{skill_block}\n\n{content}"
                    replaced = True
                elif isinstance(content, list):
                    cloned["content"] = [{"type": "text", "text": f"{skill_block}\n\n"}] + content
                    replaced = True
            updated.append(cloned)
        if replaced:
            return updated
        return prompt

    if isinstance(prompt, str):
        return f"{skill_block}\n\n{prompt}"

    return prompt


def split_reasoning_steps(completion_text: str, max_steps: int = 6) -> list[str]:
    text = strip_thinking_tags(completion_text)
    if not text:
        return []

    text = text.replace("\r\n", "\n")
    numbered = re.split(r"\n(?=\s*(?:\d+\.\s+|[-*]\s+))", text)
    candidates = numbered if len(numbered) > 1 else re.split(r"\n\s*\n", text)

    steps = []
    for chunk in candidates:
        cleaned = squeeze_whitespace(chunk.replace("\n", " "))
        if not cleaned:
            continue
        steps.append(cleaned)
        if len(steps) >= max_steps:
            break

    if not steps:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for chunk in sentences:
            cleaned = squeeze_whitespace(chunk)
            if cleaned:
                steps.append(cleaned)
            if len(steps) >= max_steps:
                break

    return steps


def summarize_step_action(step_text: str) -> str:
    text = squeeze_whitespace(step_text)
    if not text:
        return "Advance the derivation"

    lower = text.lower()
    if "case" in lower:
        return "Split into cases and analyze each branch"
    if "let " in lower or "denote" in lower:
        return "Introduce variables and rewrite the problem"
    if "therefore" in lower or "thus" in lower or "hence" in lower:
        return "Synthesize prior derivations into the next claim"
    if "=" in text or "\\frac" in text:
        return "Derive algebraic relations and simplify"
    if "angle" in lower or "triangle" in lower or "circle" in lower:
        return "Apply geometric relations to constrain the configuration"
    return text[:120]


def default_planning_pattern(completion_text: str) -> str:
    text = strip_thinking_tags(completion_text).lower()
    if "case" in text:
        return "Parse problem -> derive key relations -> split into cases -> eliminate invalid cases -> verify boxed final answer"
    if "angle" in text or "triangle" in text or "circle" in text:
        return "Parse geometric constraints -> derive structural relations -> compute target quantity -> verify boxed final answer"
    return "Parse problem -> derive mathematical relations -> compute target quantity -> verify boxed final answer"


def infer_failure_mode(completion_text: str, predicted_answer: str | None, ground_truth: str) -> tuple[str, str]:
    stripped = strip_thinking_tags(completion_text)
    if predicted_answer is None:
        return (
            "When a long derivation never resolves into a verifiable final answer format",
            "Reasoned through the problem but failed to state a final answer inside \\boxed{} for verification",
        )
    if not stripped:
        return (
            "When the attempt is incomplete or collapses before the core derivation is finished",
            "Stopped before building a complete derivation and committed no usable final conclusion",
        )
    if len(stripped) < 120:
        return (
            "When the model jumps too quickly from setup to a final claim",
            "Committed to a final answer without enough intermediate derivation to verify the conclusion",
        )
    return (
        "When a multistep derivation could contain an algebraic, arithmetic, or logical slip",
        f"Committed to the incorrect final answer {predicted_answer!r} without fully checking it against the target conditions and ground truth {ground_truth!r}",
    )
