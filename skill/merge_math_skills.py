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


GENERAL_MERGE_PROMPT = "merge_general_skills_prompt.txt"
MISTAKE_MERGE_PROMPT = "merge_common_mistakes_prompt.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchically merge raw math skill candidates into a deduplicated skill bank."
    )
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--llm_backend", default="vllm", choices=["azure", "local", "vllm"])
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--merge_group_size", type=int, default=32)
    parser.add_argument(
        "--merge_stagnation_patience",
        type=int,
        default=3,
        help=(
            "Stop merging a collection after this many consecutive layers do not reduce "
            "the skill item count. Set to 0 to disable the safeguard."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--local_device_map", default="auto")
    parser.add_argument("--local_dtype", default="bfloat16")
    parser.add_argument("--local_attn_implementation", default="flash_attention_2")
    parser.add_argument("--local_temperature", type=float, default=0.0)
    parser.add_argument("--local_top_p", type=float, default=1.0)
    parser.add_argument("--local_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--local_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--local_max_model_len", type=int, default=None)
    return parser.parse_args()


def load_skill_bank(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compact_whitespace(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def parse_response_to_json(text: str) -> dict[str, Any]:
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start == -1 or json_end <= json_start:
        raise ValueError("Model response does not contain a JSON object.")
    return json.loads(text[json_start:json_end])


def normalize_general_skill(item: dict[str, Any]) -> dict[str, Any] | None:
    title = compact_whitespace(item.get("title"))
    principle = compact_whitespace(item.get("principle"))
    when_to_apply = compact_whitespace(item.get("when_to_apply"))
    if not title or not principle or not when_to_apply:
        return None
    return {
        "skill_id": compact_whitespace(item.get("skill_id")) or "gen_tmp",
        "title": title,
        "principle": principle,
        "when_to_apply": when_to_apply,
    }


def normalize_common_mistake(item: dict[str, Any]) -> dict[str, Any] | None:
    description = compact_whitespace(item.get("description"))
    why_it_happens = compact_whitespace(item.get("why_it_happens"))
    how_to_avoid = compact_whitespace(item.get("how_to_avoid"))
    if not description or not why_it_happens or not how_to_avoid:
        return None
    return {
        "mistake_id": compact_whitespace(item.get("mistake_id")) or "err_tmp",
        "description": description,
        "why_it_happens": why_it_happens,
        "how_to_avoid": how_to_avoid,
    }


def serialize_general_skills_for_merge(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "title": item.get("title"),
            "principle": item.get("principle"),
            "when_to_apply": item.get("when_to_apply"),
        }
        for item in items
    ]


def serialize_common_mistakes_for_merge(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "description": item.get("description"),
            "why_it_happens": item.get("why_it_happens"),
            "how_to_avoid": item.get("how_to_avoid"),
        }
        for item in items
    ]


def dedupe_exact_items(items: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        signature = tuple(compact_whitespace(item.get(key)).lower() for key in keys)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
    return deduped


def chunk_list(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def merge_collection(
    *,
    client,
    items: list[dict[str, Any]],
    prompt_name: str,
    output_key: str,
    serialize_items,
    normalize_item,
    batch_size: int,
    group_size: int,
    stagnation_patience: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not items:
        return [], []

    current_items = list(items)
    layer_stats: list[dict[str, Any]] = []
    layer_index = 0
    stagnant_layers = 0

    while True:
        groups = chunk_list(current_items, group_size)
        print(
            f"Merging {output_key} layer {layer_index}: "
            f"{len(current_items)} items across {len(groups)} groups"
        )

        next_items: list[dict[str, Any]] = []
        pending_groups: list[tuple[int, list[dict[str, Any]]]] = []
        pending_prompts: list[str] = []

        for group_index, group in enumerate(groups):
            if len(group) <= 1:
                next_items.extend(group)
                continue
            pending_groups.append((group_index, group))
            pending_prompts.append(
                render_prompt(
                    prompt_name,
                    items_json=json.dumps(serialize_items(group), ensure_ascii=False, indent=2),
                )
            )

        parsed_groups: dict[int, list[dict[str, Any]]] = {}
        for start in range(0, len(pending_prompts), batch_size):
            batch_prompts = pending_prompts[start : start + batch_size]
            batch_outputs = client.batch_complete(batch_prompts)
            batch_groups = pending_groups[start : start + batch_size]
            if len(batch_outputs) != len(batch_groups):
                raise RuntimeError(
                    f"Expected {len(batch_groups)} merge responses, got {len(batch_outputs)}."
                )

            for (group_index, original_group), response in zip(batch_groups, batch_outputs):
                try:
                    parsed = parse_response_to_json(response)
                    normalized_items = [
                        normalized
                        for normalized in (normalize_item(item) for item in parsed.get(output_key, []))
                        if normalized is not None
                    ]
                except (ValueError, json.JSONDecodeError):
                    normalized_items = []

                parsed_groups[group_index] = normalized_items or original_group

        for group_index, group in enumerate(groups):
            next_items.extend(parsed_groups.get(group_index, group if len(group) > 1 else group))

        if len(next_items) < len(current_items):
            stagnant_layers = 0
        else:
            stagnant_layers += 1

        stopped_reason = None
        if len(groups) == 1:
            stopped_reason = "root_group_reached"
        elif stagnation_patience > 0 and stagnant_layers >= stagnation_patience:
            stopped_reason = "stagnation_patience_reached"

        layer_stats.append(
            {
                "layer_index": layer_index,
                "input_items": len(current_items),
                "num_groups": len(groups),
                "output_items": len(next_items),
                "stagnant_layers": stagnant_layers,
                "stopped_reason": stopped_reason,
            }
        )

        if stopped_reason == "root_group_reached":
            return next_items, layer_stats
        if stopped_reason == "stagnation_patience_reached":
            print(
                f"Stopping {output_key} merge after {stagnant_layers} consecutive "
                f"non-decreasing layers: {len(current_items)} -> {len(next_items)} items"
            )
            return next_items, layer_stats

        current_items = next_items
        layer_index += 1


def reassign_ids(items: list[dict[str, Any]], id_field: str, prefix: str) -> list[dict[str, Any]]:
    reassigned: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        updated = dict(item)
        updated[id_field] = f"{prefix}_{index:03d}"
        reassigned.append(updated)
    return reassigned


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer.")
    if args.merge_group_size <= 1:
        raise ValueError("--merge_group_size must be greater than 1.")
    if args.merge_stagnation_patience < 0:
        raise ValueError("--merge_stagnation_patience must be non-negative.")

    bank = load_skill_bank(args.skills_path)

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

    raw_general_skills = [
        normalized
        for normalized in (normalize_general_skill(item) for item in bank.get("general_skills", []))
        if normalized is not None
    ]
    raw_common_mistakes = [
        normalized
        for normalized in (normalize_common_mistake(item) for item in bank.get("common_mistakes", []))
        if normalized is not None
    ]

    merged_general_skills, general_stats = merge_collection(
        client=client,
        items=raw_general_skills,
        prompt_name=GENERAL_MERGE_PROMPT,
        output_key="general_skills",
        serialize_items=serialize_general_skills_for_merge,
        normalize_item=normalize_general_skill,
        batch_size=args.batch_size,
        group_size=args.merge_group_size,
        stagnation_patience=args.merge_stagnation_patience,
    )
    merged_common_mistakes, mistake_stats = merge_collection(
        client=client,
        items=raw_common_mistakes,
        prompt_name=MISTAKE_MERGE_PROMPT,
        output_key="common_mistakes",
        serialize_items=serialize_common_mistakes_for_merge,
        normalize_item=normalize_common_mistake,
        batch_size=args.batch_size,
        group_size=args.merge_group_size,
        stagnation_patience=args.merge_stagnation_patience,
    )

    merged_general_skills = reassign_ids(
        dedupe_exact_items(merged_general_skills, ("title", "principle", "when_to_apply")),
        "skill_id",
        "gen",
    )
    merged_common_mistakes = reassign_ids(
        dedupe_exact_items(merged_common_mistakes, ("description", "why_it_happens", "how_to_avoid")),
        "mistake_id",
        "err",
    )

    output = {
        "general_skills": merged_general_skills,
        "task_specific_skills": {},
        "common_mistakes": merged_common_mistakes,
        "metadata": {
            "source": f"hierarchical merge from {args.skills_path} using {args.llm_backend}:{args.model}",
            "merge_group_size": args.merge_group_size,
            "merge_stagnation_patience": args.merge_stagnation_patience,
            "batch_size": args.batch_size,
            "input_general_skills": len(raw_general_skills),
            "output_general_skills": len(merged_general_skills),
            "input_common_mistakes": len(raw_common_mistakes),
            "output_common_mistakes": len(merged_common_mistakes),
            "general_merge_layers": general_stats,
            "common_mistake_merge_layers": mistake_stats,
        },
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved merged skill library to {args.output_path}")
    print(f"Merged general skills: {len(merged_general_skills)}")
    print(f"Merged common mistakes: {len(merged_common_mistakes)}")


if __name__ == "__main__":
    main()
