from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.dataset_adapter import extract_question, extract_reward_solution, load_dataset_with_mode
from reward import make_reward_scorer

try:
    from .memory_builder import build_memory_record
    from .prompts import render_prompt
except ImportError:
    from skill.memory_builder import build_memory_record
    from skill.prompts import render_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SkillRL-style math memories from OPSD datasets.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--generation_backend", default="vllm", choices=["transformers", "vllm"])
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=38912)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=None)
    parser.add_argument("--vllm_dtype", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_generation_backend(args: argparse.Namespace):
    if args.generation_backend == "vllm":
        from vllm import LLM

        llm_kwargs: dict[str, object] = {
            "model": args.model_name_or_path,
            "trust_remote_code": True,
            "tensor_parallel_size": args.vllm_tensor_parallel_size,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "dtype": args.vllm_dtype,
        }
        if args.vllm_max_model_len is not None:
            llm_kwargs["max_model_len"] = args.vllm_max_model_len
        return LLM(**llm_kwargs)

    dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )


def generate_completions(
    *,
    backend: str,
    generator,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    if backend == "vllm":
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = generator.generate(prompts, sampling_params)
        return [output.outputs[0].text if output.outputs else "" for output in outputs]

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    input_device = next(generator.parameters()).device
    encoded = {key: value.to(input_device) for key, value in encoded.items()}

    with torch.no_grad():
        generated = generator.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion_ids = generated[:, encoded["input_ids"].shape[1] :]
    return tokenizer.batch_decode(completion_ids, skip_special_tokens=False)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path("outputs") / "cache" / "hf_datasets"))
    random.seed(args.seed)

    dataset_mode, _, dataset = load_dataset_with_mode(args.dataset_name)
    if args.dataset_split not in dataset:
        raise KeyError(f"Split {args.dataset_split!r} not found. Available: {list(dataset.keys())}")

    ds = dataset[args.dataset_split]
    if args.max_samples is not None:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.max_samples, len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generator = load_generation_backend(args)
    scorer = make_reward_scorer(mode=dataset_mode, binary_output=False)

    records = []
    for start in range(0, len(ds), args.batch_size):
        batch = ds.select(range(start, min(start + args.batch_size, len(ds))))
        prompts = []
        questions = []
        ground_truths = []
        data_sources = []

        for example in batch:
            question = extract_question(example, dataset_mode)
            prompt_text = render_prompt("memory_generation_prompt.txt", problem=question)
            messages = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            questions.append(question)
            ground_truths.append(extract_reward_solution(example, dataset_mode))
            data_sources.append(str(example.get("data_source", dataset_mode)))

        completions = generate_completions(
            backend=args.generation_backend,
            generator=generator,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        reward_items = scorer(completions, ground_truths)

        for offset, (question, completion, reward_item, ground_truth, data_source) in enumerate(
            zip(questions, completions, reward_items, ground_truths, data_sources, strict=True)
        ):
            reward_item = dict(reward_item)
            reward_item["ground_truth"] = ground_truth
            records.append(
                build_memory_record(
                    question=question,
                    completion_text=completion,
                    reward_result=reward_item,
                    dataset_mode=dataset_mode,
                    data_source=data_source,
                    origin_env_id=f"{dataset_mode}_{args.dataset_split}_{start + offset}",
                )
            )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    successes = sum(1 for record in records if record["tags"]["outcome"] == "Success")
    failures = len(records) - successes
    print(f"Saved {len(records)} memories to {args.output_path}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")


if __name__ == "__main__":
    main()
