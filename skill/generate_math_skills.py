from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .llm_client import build_chat_client
    from .prompts import render_prompt
except ImportError:
    from skill.llm_client import build_chat_client
    from skill.prompts import render_prompt


SUCCESS_PROMPT = "success_skill_from_memory_prompt.txt"
FAILURE_PROMPT = "failure_skill_from_memory_prompt.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract raw math skill candidates from memories with one memory per LLM call."
    )
    parser.add_argument("--memory_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--llm_backend", default="vllm", choices=["azure", "local", "vllm"])
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--local_device_map", default="auto")
    parser.add_argument("--local_dtype", default="bfloat16")
    parser.add_argument("--local_attn_implementation", default="flash_attention_2")
    parser.add_argument("--local_temperature", type=float, default=0.0)
    parser.add_argument("--local_top_p", type=float, default=1.0)
    parser.add_argument("--local_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--local_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--local_max_model_len", type=int, default=None)
    return parser.parse_args()


def load_memories(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compact_whitespace(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def truncate_text(text: Any, max_chars: int) -> str:
    normalized = compact_whitespace(text)
    if not normalized or len(normalized) <= max_chars:
        return normalized
    head = max_chars // 2
    tail = max_chars - head - len(" ...[snip]... ")
    tail = max(tail, 0)
    return f"{normalized[:head].rstrip()} ...[snip]... {normalized[-tail:].lstrip()}"


def compress_memory(memory: dict[str, Any]) -> dict[str, Any]:
    content = memory.get("content", {})
    task_meta = content.get("task_meta", {})
    strategic = content.get("strategic_guidelines", {})
    if "strategic_guidelines" in strategic:
        strategic = strategic["strategic_guidelines"]
    raw_attempt = content.get("raw_attempt", {})
    outcome = memory.get("tags", {}).get("outcome") or "Failure"

    compressed: dict[str, Any] = {
        "memory_id": memory.get("memory_id"),
        "goal": task_meta.get("original_goal"),
        "outcome": outcome,
        "data_source": memory.get("tags", {}).get("data_source"),
        "planning_pattern": strategic.get("planning_pattern"),
        "predicted_answer": raw_attempt.get("predicted_answer"),
        "ground_truth": raw_attempt.get("ground_truth") or task_meta.get("ground_truth"),
        "reward": raw_attempt.get("reward"),
        "feedback": compact_whitespace(raw_attempt.get("feedback")),
        "completion_excerpt": truncate_text(
            raw_attempt.get("completion"),
            max_chars=1600 if outcome == "Success" else 2200,
        ),
    }

    if outcome == "Success":
        compressed["refined_trajectory"] = content.get("refined_trajectory") or []
    else:
        compressed["mistakes_to_avoid"] = strategic.get("mistakes_to_avoid") or []

    return compressed


def render_memory_prompt(memory: dict[str, Any]) -> str:
    prompt_name = SUCCESS_PROMPT if memory.get("outcome") == "Success" else FAILURE_PROMPT
    return render_prompt(
        prompt_name,
        memory_json=json.dumps(memory, ensure_ascii=False, indent=2),
    )


def parse_response_to_json(text: str) -> dict[str, Any]:
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start == -1 or json_end <= json_start:
        raise ValueError("Model response does not contain a JSON object.")
    return json.loads(text[json_start:json_end])


def normalize_general_skill(item: dict[str, Any], skill_id: str) -> dict[str, Any] | None:
    title = compact_whitespace(item.get("title"))
    principle = compact_whitespace(item.get("principle"))
    when_to_apply = compact_whitespace(item.get("when_to_apply"))
    if not title or not principle or not when_to_apply:
        return None
    return {
        "skill_id": skill_id,
        "title": title,
        "principle": principle,
        "when_to_apply": when_to_apply,
    }


def normalize_common_mistake(item: dict[str, Any], mistake_id: str) -> dict[str, Any] | None:
    description = compact_whitespace(item.get("description"))
    why_it_happens = compact_whitespace(item.get("why_it_happens"))
    how_to_avoid = compact_whitespace(item.get("how_to_avoid"))
    if not description or not why_it_happens or not how_to_avoid:
        return None
    return {
        "mistake_id": mistake_id,
        "description": description,
        "why_it_happens": why_it_happens,
        "how_to_avoid": how_to_avoid,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer.")

    memories = load_memories(args.memory_path)
    compressed_memories = [compress_memory(memory) for memory in memories]

    client = build_chat_client(
        backend=args.llm_backend,
        model=args.model,
        max_completion_tokens=args.max_new_tokens,
        local_device_map=args.local_device_map,
        local_dtype=args.local_dtype,
        local_attn_implementation=args.local_attn_implementation,
        local_temperature=args.local_temperature,
        local_top_p=args.local_top_p,
        local_tensor_parallel_size=args.local_tensor_parallel_size,
        local_gpu_memory_utilization=args.local_gpu_memory_utilization,
        local_max_model_len=args.local_max_model_len,
    )

    general_skills: list[dict[str, Any]] = []
    common_mistakes: list[dict[str, Any]] = []
    next_general_idx = 1
    next_mistake_idx = 1
    parse_failures = 0

    total_batches = (len(compressed_memories) + args.batch_size - 1) // args.batch_size if compressed_memories else 0
    for batch_index, start in enumerate(range(0, len(compressed_memories), args.batch_size), start=1):
        batch_memories = compressed_memories[start : start + args.batch_size]
        prompts = [render_memory_prompt(memory) for memory in batch_memories]
        responses = client.batch_complete(prompts)

        if len(responses) != len(batch_memories):
            raise RuntimeError(
                f"Expected {len(batch_memories)} responses for batch {batch_index}, got {len(responses)}."
            )

        for memory, response in zip(batch_memories, responses):
            try:
                parsed = parse_response_to_json(response)
            except (ValueError, json.JSONDecodeError):
                parse_failures += 1
                continue

            if memory.get("outcome") == "Success":
                raw_items = parsed.get("general_skills", [])[:3]
                for raw_item in raw_items:
                    skill = normalize_general_skill(raw_item, f"gen_cand_{next_general_idx:06d}")
                    if skill is None:
                        continue
                    general_skills.append(skill)
                    next_general_idx += 1
            else:
                raw_items = parsed.get("common_mistakes", [])[:3]
                for raw_item in raw_items:
                    mistake = normalize_common_mistake(raw_item, f"err_cand_{next_mistake_idx:06d}")
                    if mistake is None:
                        continue
                    common_mistakes.append(mistake)
                    next_mistake_idx += 1

        print(
            f"Processed skill-candidate batch {batch_index}/{total_batches}: "
            f"{min(start + len(batch_memories), len(compressed_memories))}/{len(compressed_memories)} memories"
        )

    success_count = sum(1 for memory in compressed_memories if memory.get("outcome") == "Success")
    failure_count = len(compressed_memories) - success_count
    output = {
        "general_skills": general_skills,
        "task_specific_skills": {},
        "common_mistakes": common_mistakes,
        "metadata": {
            "source": f"single-memory extraction from {args.memory_path} using {args.llm_backend}:{args.model}",
            "generation_mode": "single_memory_parallel_candidates",
            "merge_recommended": True,
            "total_memories_processed": len(compressed_memories),
            "success_memories": success_count,
            "failure_memories": failure_count,
            "batch_size": args.batch_size,
            "parse_failures": parse_failures,
        },
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved raw skill candidates to {args.output_path}")
    print(f"General skill candidates: {len(output['general_skills'])}")
    print(f"Common mistake candidates: {len(output['common_mistakes'])}")
    if parse_failures:
        print(f"Skipped {parse_failures} memories because the model response could not be parsed as JSON.")


if __name__ == "__main__":
    main()
