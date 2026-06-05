import sys
import torch
import argparse
import json
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_verify import parse, verify
from skill.skill_memory import MathSkillsOnlyMemory


def extract_boxed_answer(text: str) -> str:
    """
    Extract answer from \\boxed{} command in the text.
    Returns the last boxed answer found.
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    right_brace_idx = None

    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    boxed_str = text[idx : right_brace_idx + 1]

    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        answer = boxed_str[7:-1]
        return answer.strip()

    return None


def grade_answer(predicted: str, ground_truth: str) -> bool:
    """
    Grade the predicted answer against ground truth using math_verify.

    Args:
        predicted: The predicted answer (already extracted from \\boxed{})
        ground_truth: The ground truth answer

    Returns:
        True if answers match, False otherwise
    """
    if predicted is None:
        return False

    try:
        if not "$" in predicted:
            predicted = f"${predicted}$"
        if not "$" in ground_truth:
            ground_truth = f"${ground_truth}$"

        pred_parsed = parse(predicted, fallback_mode="no_fallback")
        gt_parsed = parse(ground_truth, fallback_mode="no_fallback")

        return verify(gt_parsed, pred_parsed, timeout_seconds=5)
    except Exception:
        pred_norm = predicted.replace("$", "").replace(" ", "").lower().strip()
        gt_norm = ground_truth.replace("$", "").replace(" ", "").lower().strip()
        return pred_norm == gt_norm


def load_vllm_model(
    base_model_path: str,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int = None,
    enable_thinking: bool = True,
):
    """
    Load a model using vLLM for fast inference.

    Args:
        base_model_path: Path to the base model
        lora_adapter_path: Path to the LoRA adapters (checkpoint directory). If None, uses base model only.
        gpu_memory_utilization: GPU memory utilization (0.0 to 1.0)
        tensor_parallel_size: Number of GPUs to use for tensor parallelism
        max_model_len: Maximum model context length
        enable_thinking: Whether to enable thinking mode for Qwen3

    Returns:
        Tuple of (vLLM LLM instance, tokenizer)
    """
    print(f"Loading model with vLLM from: {base_model_path}")

    if max_model_len is None:
        max_model_len = 40960 if enable_thinking else 32768
        print(
            f"Auto-setting max_model_len to {max_model_len} for {'thinking' if enable_thinking else 'non-thinking'} mode"
        )

    llm_config = {
        "model": base_model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "distributed_executor_backend": "mp",
        "enforce_eager": True,
    }

    if lora_adapter_path is not None:
        print(f"LoRA adapter path provided: {lora_adapter_path}")

        adapter_path = Path(lora_adapter_path) / "adapter_model.safetensors"
        if not adapter_path.exists():
            adapter_path = Path(lora_adapter_path) / "adapter_model.bin"

        if adapter_path.exists():
            print("LoRA weights found. Enabling LoRA support...")
            llm_config["enable_lora"] = True
            llm_config["max_lora_rank"] = 64
            llm_config["max_loras"] = 1
            llm_config["max_cpu_loras"] = 1
        else:
            print(f"Warning: No LoRA weights found at {lora_adapter_path}")
            print("Continuing with base model only...")
            lora_adapter_path = None

    llm = LLM(**llm_config)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print("vLLM model loaded successfully!")
    return llm, tokenizer


def load_dataset_with_retry(dataset_path: str, split: str, **kwargs):
    """
    Load a Hugging Face dataset and automatically retry with force_redownload
    if the local cache metadata exists but the underlying Arrow files are missing.
    """
    try:
        return load_dataset(dataset_path, split=split, **kwargs)
    except FileNotFoundError as exc:
        print("\n" + "!" * 70)
        print("Detected broken Hugging Face dataset cache.")
        print(f"Dataset: {dataset_path} | split: {split}")
        print(f"Original error: {exc}")
        print("Retrying with download_mode='force_redownload' ...")
        print("!" * 70 + "\n")
        return load_dataset(dataset_path, split=split, download_mode="force_redownload", **kwargs)


def build_skill_memory(
    skills_json_path: str | None,
    retrieval_mode: str = "embedding",
    embedding_model_path: str = "Qwen/Qwen3-Embedding-0.6B",
    random_seed: int = 42,
):
    if not skills_json_path:
        return None
    return MathSkillsOnlyMemory(
        skills_json_path=skills_json_path,
        retrieval_mode=retrieval_mode,
        embedding_model_path=embedding_model_path,
        random_seed=random_seed,
    )


def build_skill_augmented_problem(problem: str, skill_memory, skills_top_k: int) -> tuple[str, str]:
    if skill_memory is None:
        return f"Problem: {problem}", ""

    retrieved = skill_memory.retrieve(task_description=problem, top_k=skills_top_k)
    skill_block = skill_memory.format_for_prompt(retrieved)
    if not skill_block:
        return f"Problem: {problem}", ""
    return f"{skill_block}\n\nProblem: {problem}", skill_block


def evaluate_math500(
    llm,
    tokenizer,
    max_new_tokens: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    num_samples: int = None,
    output_file: str = None,
    lora_request=None,
    dataset_name: str = "math500",
    base_model_name: str = None,
    enable_thinking: bool = True,
    val_n: int = 1,
    skill_memory=None,
    skills_top_k: int = 8,
    skills_json_path: str | None = None,
    skills_retrieval_mode: str | None = None,
):
    """
    Evaluate model on MATH500 or other datasets using Qwen3 thinking mode with best practices.

    Args:
        llm: The vLLM LLM instance
        tokenizer: The tokenizer for chat template
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.6 for thinking, 0.7 for non-thinking)
        top_p: Top-p sampling parameter (0.95 for thinking, 0.8 for non-thinking)
        top_k: Top-k sampling parameter (20 recommended)
        min_p: Minimum probability threshold (0 recommended)
        presence_penalty: Presence penalty to reduce repetitions (0-2)
        num_samples: Number of samples to evaluate (None = all)
        output_file: Path to save detailed results
        lora_request: Optional LoRA request for inference
        dataset_name: Name of dataset to use
        base_model_name: Base model name for logging
        enable_thinking: Whether to use thinking mode
    """
    print(f"Loading {dataset_name.upper()} dataset...")
    if dataset_name.lower() == "math500":
        dataset = load_dataset_with_retry("HuggingFaceH4/MATH-500", split="test")
        print(f"Loaded HuggingFaceH4/MATH-500 dataset with {len(dataset)} problems")
    elif dataset_name.lower() == "amo-bench":
        dataset = load_dataset_with_retry("meituan-longcat/AMO-Bench", split="test")
        print(f"Loaded meituan-longcat/AMO-Bench dataset with {len(dataset)} problems")
    elif dataset_name.lower() == "minerva":
        dataset = load_dataset_with_retry("math-ai/minervamath", split="test")
        print(f"Loaded minerva dataset with {len(dataset)} problems")
    elif dataset_name.lower() == "amc23":
        dataset = load_dataset_with_retry("math-ai/amc23", split="test")
        print(f"Loaded amc 23 dataset with {len(dataset)} problems")
    elif dataset_name.lower() == "aime24":
        dataset = load_dataset_with_retry("HuggingFaceH4/aime_2024", split="train")
        print(f"Loaded HuggingFaceH4/aime_2024 dataset with {len(dataset)} problems")
    elif dataset_name.lower() == "aime25":
        dataset = load_dataset_with_retry("yentinglin/aime_2025", split="train", trust_remote_code=True)
        print(f"Loaded yentinglin/aime_2025 dataset with {len(dataset)} problems")
    elif dataset_name.lower() == "hmmt25":
        dataset = load_dataset_with_retry("MathArena/hmmt_feb_2025", split="train", trust_remote_code=True)
        print(f"Loaded MathArena/hmmt_feb_2025 dataset with {len(dataset)} problems")
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose 'math500', 'amo-bench', 'aime24', 'aime25', 'hmmt25', 'minerva', or 'amc23'"
        )

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} problems with vLLM batch inference...")

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_new_tokens,
        presence_penalty=presence_penalty,
        n=val_n,
    )

    total = 0
    formatted_count = 0
    results = []

    pass_at_n = 0
    total_correct_per_problem = 0

    all_messages = []
    all_gt_answers = []
    all_problems = []
    all_question_ids = []
    all_skill_blocks = []

    for example in dataset:
        if dataset_name.lower() == "amo-bench":
            problem = example["prompt"]
            gt_answer = example["answer"]
            question_id = example.get("question_id", None)
        elif dataset_name.lower() == "aime24":
            problem = example["problem"]
            gt_answer = example["answer"]
            question_id = example.get("id", None)
        elif dataset_name.lower() == "minerva":
            problem = example["question"]
            gt_answer = example["answer"]
            question_id = example.get("id", None)
        elif dataset_name.lower() == "amc23":
            problem = example["question"]
            gt_answer = example["answer"]
            question_id = example.get("id", None)
        elif dataset_name.lower() == "aime25":
            problem = example["problem"]
            gt_answer = str(example["answer"])
            question_id = example.get("problem_idx", None)
        elif dataset_name.lower() == "hmmt25":
            problem = example["problem"]
            gt_answer = str(example["answer"])
            question_id = example.get("problem_idx", None)
        else:
            problem = example["problem"]
            gt_solution = example["solution"]
            question_id = None
            gt_answer = extract_boxed_answer(gt_solution)
            if gt_answer is None:
                gt_answer = gt_solution

        prompt_body, injected_skill_block = build_skill_augmented_problem(problem, skill_memory, skills_top_k)
        user_message = (
            f"{prompt_body}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        )

        messages = [{"role": "user", "content": user_message}]

        all_messages.append(messages)
        all_gt_answers.append(gt_answer)
        all_problems.append(problem)
        all_question_ids.append(question_id)
        all_skill_blocks.append(injected_skill_block)

    print(f"\nRunning vLLM batch inference on {len(all_messages)} problems...")

    all_prompts = []
    for messages in all_messages:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        all_prompts.append(text)

    if lora_request is not None:
        if lora_request.lora_path is None:
            raise ValueError("LoRA request requires a non-empty lora_local_path.")

    if lora_request is not None:
        outputs = llm.generate(all_prompts, sampling_params, lora_request=lora_request, use_tqdm=True)
    else:
        outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

    print("\nProcessing results...")
    for idx, (output, problem, gt_answer, question_id, skill_block) in enumerate(
        zip(outputs, all_problems, all_gt_answers, all_question_ids, all_skill_blocks)
    ):
        generations = []
        predicted_answers = []
        is_correct_list = []
        is_formatted_list = []

        for i in range(len(output.outputs)):
            generated_text = output.outputs[i].text

            predicted_answer = extract_boxed_answer(generated_text)
            is_formatted = predicted_answer is not None
            is_correct = grade_answer(predicted_answer, gt_answer)

            generations.append(generated_text)
            predicted_answers.append(predicted_answer if predicted_answer else "[No boxed answer found]")
            is_correct_list.append(is_correct)
            is_formatted_list.append(is_formatted)

        num_correct = sum(is_correct_list)
        num_formatted = sum(is_formatted_list)
        has_correct = any(is_correct_list)

        majority_vote_correct = False
        if num_formatted > 0:
            from collections import Counter

            formatted_predictions = [pred for pred, fmt in zip(predicted_answers, is_formatted_list) if fmt]
            if formatted_predictions:
                most_common_answer = Counter(formatted_predictions).most_common(1)[0][0]
                majority_vote_correct = grade_answer(most_common_answer, gt_answer)

        if has_correct:
            pass_at_n += 1
        total_correct_per_problem += num_correct
        formatted_count += num_formatted
        total += val_n

        result = {
            "problem_id": question_id if question_id is not None else idx,
            "problem": problem,
            "ground_truth": gt_answer,
            "used_skills": bool(skill_block),
            "retrieved_skill_block": skill_block,
            "val_n": val_n,
            "generations": [
                {"predicted_answer": pred, "full_generation": gen, "correct": corr, "formatted": fmt}
                for pred, gen, corr, fmt in zip(
                    predicted_answers, generations, is_correct_list, is_formatted_list
                )
            ],
            "num_correct": num_correct,
            "pass_at_n": has_correct,
            "majority_vote_correct": majority_vote_correct,
            "predicted_answer": predicted_answers[0],
            "full_generation": generations[0],
            "correct": is_correct_list[0],
            "formatted": is_formatted_list[0],
        }
        results.append(result)

        format_rate = formatted_count / total * 100
        current_pass_at_n = pass_at_n / (idx + 1) * 100
        current_avg_at_n = total_correct_per_problem / total * 100

        status = "PASS" if has_correct else "FAIL"
        print(
            f"{status} [{idx + 1}/{len(dataset)}] Pass@{val_n}: {current_pass_at_n:.1f}% | Avg@{val_n}: {current_avg_at_n:.1f}% | Formatted: {format_rate:.1f}%"
        )

    num_problems = len(dataset)
    format_rate = formatted_count / total * 100

    pass_at_n_pct = pass_at_n / num_problems * 100
    average_at_n_pct = total_correct_per_problem / total * 100

    majority_vote_correct_count = sum(1 for r in results if r["majority_vote_correct"])
    majority_vote_at_n_pct = majority_vote_correct_count / num_problems * 100

    print("\n" + "=" * 70)
    print(f"FINAL RESULTS")
    print("=" * 70)
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Thinking Mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Total problems: {num_problems}")
    print(f"Solutions per problem: {val_n}")
    print(f"Total solutions: {total}")
    print(f"\nMetrics:")
    print(f"  Pass@{val_n}: {pass_at_n_pct:.2f}% ({pass_at_n}/{num_problems})")
    print(f"  Average@{val_n}: {average_at_n_pct:.2f}% ({total_correct_per_problem}/{total})")
    print(
        f"  Majority Vote@{val_n}: {majority_vote_at_n_pct:.2f}% ({majority_vote_correct_count}/{num_problems})"
    )
    print(f"\nFormatting:")
    print(f"  Formatted (boxed) answers: {formatted_count}/{total}")
    print(f"  Format rate: {format_rate:.2f}%")
    print("=" * 70)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "base_model": base_model_name,
            "dataset": dataset_name,
            "skills_enabled": skill_memory is not None,
            "skills_json_path": skills_json_path,
            "skills_retrieval_mode": skills_retrieval_mode,
            "skills_top_k": skills_top_k if skill_memory is not None else None,
            "enable_thinking": enable_thinking,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "max_new_tokens": max_new_tokens,
            "val_n": val_n,
            "num_problems": num_problems,
            "total_solutions": total,
            "pass_at_n": pass_at_n,
            "pass_at_n_pct": pass_at_n_pct,
            "average_at_n": total_correct_per_problem,
            "average_at_n_pct": average_at_n_pct,
            "majority_vote_at_n": majority_vote_correct_count,
            "majority_vote_at_n_pct": majority_vote_at_n_pct,
            "formatted_count": formatted_count,
            "format_rate": format_rate,
            "results": results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to: {output_file}")

    return average_at_n_pct, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on MATH tasks with Qwen3 thinking mode")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Base model path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to checkpoint directory with LoRA adapters. If not provided, will use base model only.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        choices=["math500", "amo-bench", "aime24", "aime25", "hmmt25", "minerva", "amc23"],
        help="Dataset to use for evaluation (default: math500)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=38912,
        help="Maximum tokens to generate (default: 32768, use 38912 for complex competition problems)",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=True,
        help="Enable Qwen3 thinking mode (default: True)",
    )
    parser.add_argument(
        "--no_thinking", dest="enable_thinking", action="store_false", help="Disable Qwen3 thinking mode"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (auto: 0.6 for thinking, 0.7 for non-thinking)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p sampling parameter (auto: 0.95 for thinking, 0.8 for non-thinking)",
    )
    parser.add_argument(
        "--top_k", type=int, default=-1, help="Top-k sampling parameter (default: -1, disabled)"
    )
    parser.add_argument(
        "--min_p", type=float, default=0.0, help="Minimum probability threshold (default: 0.0)"
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Presence penalty to reduce repetitions (0-2, default: 0.0)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Number of samples to evaluate (None = all)"
    )
    parser.add_argument("--output_file", type=str, default=None, help="Path to save detailed results JSON")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (0.0 to 1.0, default: 0.9)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model context length (auto: 40960 for thinking, 32768 for non-thinking)",
    )
    parser.add_argument(
        "--val_n", type=int, default=6, help="Number of solutions to sample per problem (default: 6)"
    )
    parser.add_argument(
        "--skills_json_path",
        type=str,
        default=None,
        help="Optional path to a skill bank JSON. If provided, retrieved skills are injected into each eval prompt.",
    )
    parser.add_argument(
        "--skills_retrieval_mode",
        type=str,
        default="embedding",
        choices=["embedding", "random"],
        help="Skill retrieval mode to use during evaluation.",
    )
    parser.add_argument(
        "--skills_embedding_model_path",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model path used when skills_retrieval_mode=embedding.",
    )
    parser.add_argument(
        "--skills_top_k",
        type=int,
        default=8,
        help="Number of general skills and common mistakes to retrieve into each prompt.",
    )
    parser.add_argument(
        "--skills_random_seed", type=int, default=42, help="Seed used for deterministic random skill retrieval."
    )

    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        checkpoint_path = Path(args.checkpoint_dir)
        if not checkpoint_path.exists():
            print(f"\n{'='*70}")
            print("ERROR: Checkpoint directory does not exist")
            print(f"{'='*70}")
            print(f"Provided checkpoint directory: {args.checkpoint_dir}")
            print("This directory does not exist.")
            print(
                "\nPlease provide a valid checkpoint directory or omit --checkpoint_dir to use the base model only."
            )
            print(f"{'='*70}\n")
            exit(1)

    if args.skills_json_path is not None:
        skills_path = Path(args.skills_json_path)
        if not skills_path.exists():
            print("\n" + "=" * 70)
            print("ERROR: Skills JSON does not exist")
            print("=" * 70)
            print(f"Provided skills JSON: {args.skills_json_path}")
            print("Please provide a valid skill bank path or omit --skills_json_path.")
            print("=" * 70 + "\n")
            exit(1)
        if args.skills_top_k <= 0:
            print("ERROR: --skills_top_k must be positive when skills are enabled.")
            exit(1)

    if args.top_p is None:
        args.top_p = 0.95 if args.enable_thinking else 0.8
        print(
            f"Auto-setting top_p to {args.top_p} for {'thinking' if args.enable_thinking else 'non-thinking'} mode"
        )

    if args.enable_thinking and args.temperature == 0.0:
        print("\n" + "!" * 70)
        print("WARNING: Using greedy decoding (temperature=0.0) in thinking mode!")
        print("Qwen3 recommends temperature=0.6 for thinking mode to avoid")
        print("performance degradation and endless repetitions.")
        print("!" * 70 + "\n")

    if args.output_file is None:
        parts = ["eval_results", args.dataset, Path(args.base_model).name]
        if args.checkpoint_dir:
            checkpoint_path = Path(args.checkpoint_dir)
            parts += [checkpoint_path.parent.name, checkpoint_path.name]
        else:
            parts += ["base"]
        if args.skills_json_path:
            parts += [f"skills-{Path(args.skills_json_path).stem}", args.skills_retrieval_mode, f"topk{args.skills_top_k}"]
        parts += [
            "thinking" if args.enable_thinking else "nonthinking",
            f"temp{args.temperature}",
            f"valn{args.val_n}",
        ]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = str(Path("eval_results") / ("_".join(parts) + f"_{timestamp}.json"))

    print(f"Results will be saved to: {args.output_file}")

    print("\n" + "=" * 70)
    print("QWEN3 MATH EVALUATION WITH THINKING MODE")
    print("=" * 70)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint_dir or 'None (base model only)'}")
    print(f"Skills: {args.skills_json_path or 'None'}")
    if args.skills_json_path:
        print(f"Skill retrieval mode: {args.skills_retrieval_mode}")
        print(f"Skill/mistake top-k: {args.skills_top_k}")
    print(f"Thinking Mode: {'ENABLED' if args.enable_thinking else 'DISABLED'}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(
        f"Temperature: {args.temperature} (Qwen3 {'thinking' if args.enable_thinking else 'non-thinking'} mode)"
    )
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Min-p: {args.min_p}")
    print(f"Presence penalty: {args.presence_penalty}")
    print(f"Num samples: {args.num_samples or 'All'}")
    print(f"Val-N (solutions per problem): {args.val_n}")
    print(f"Output file: {args.output_file}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print("=" * 70 + "\n")

    skill_memory = build_skill_memory(
        skills_json_path=args.skills_json_path,
        retrieval_mode=args.skills_retrieval_mode,
        embedding_model_path=args.skills_embedding_model_path,
        random_seed=args.skills_random_seed,
    )

    llm, tokenizer = load_vllm_model(
        args.base_model,
        args.checkpoint_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
    )

    lora_request = None
    if args.checkpoint_dir is not None:
        try:
            from vllm.lora.request import LoRARequest

            adapter_safetensors = Path(args.checkpoint_dir) / "adapter_model.safetensors"
            adapter_bin = Path(args.checkpoint_dir) / "adapter_model.bin"

            if adapter_safetensors.exists() or adapter_bin.exists():
                lora_request = LoRARequest("checkpoint_lora", 1, args.checkpoint_dir)
                print(f"Successfully created LoRA request for: {args.checkpoint_dir}")
            else:
                print(f"Warning: No LoRA adapter weights found at {args.checkpoint_dir}")
                print("Expected 'adapter_model.safetensors' or 'adapter_model.bin'")
                print("Continuing with base model only...")
        except ImportError:
            print("Warning: Could not import LoRARequest. Running without LoRA.")
        except Exception as e:
            print(f"Warning: Could not create LoRA request: {e}")
            print("Continuing without LoRA.")

    average_at_n_pct, results = evaluate_math500(
        llm,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        num_samples=args.num_samples,
        output_file=args.output_file,
        lora_request=lora_request,
        dataset_name=args.dataset,
        base_model_name=args.base_model,
        enable_thinking=args.enable_thinking,
        val_n=args.val_n,
        skill_memory=skill_memory,
        skills_top_k=args.skills_top_k,
        skills_json_path=args.skills_json_path,
        skills_retrieval_mode=args.skills_retrieval_mode if args.skills_json_path else None,
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"Final Average@{args.val_n}: {average_at_n_pct:.2f}%")
    print(f"Results saved to: {args.output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
