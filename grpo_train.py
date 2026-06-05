from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import wandb
from transformers import AutoTokenizer
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from data.dataset_adapter import load_dataset_with_mode, make_grpo_format_prompt
from reward import make_reward_function
from skill.grpo_trainer import SkillAwareGRPOTrainer
from skill.training import SkillRetrievalArguments, build_skill_runtime, make_grpo_raw_prompt_example
from train_resume import find_latest_checkpoint


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    dataset_name: str = field(
        default="BytedTsinghua-SIA/DAPO-Math-17k",
        metadata={"help": "Dataset name or path passed to datasets.load_dataset()."},
    )
    dataset_split: str = field(default="train", metadata={"help": "Dataset split used for training."})
    run_config: str | None = field(
        default=None,
        metadata={"help": "Run name used for the output directory suffix and W&B run name."},
    )
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": "W&B entity used for logging."},
    )
    wandb_project: str = field(
        default="grpo-training",
        metadata={"help": "W&B project used for logging."},
    )
    reward_mode: str = field(
        default="dapo-math",
        metadata={"help": "Reward mode. Supported modes: 'openthought' and 'dapo-math'."},
    )
    reward_binary_output: bool = field(
        default=False,
        metadata={"help": "Map routed reward scores to binary values."},
    )


def _resolve_model_dtype(model_args: ModelConfig) -> torch.dtype:
    dtype = getattr(model_args, "torch_dtype", None)
    if dtype is None:
        dtype = getattr(model_args, "dtype", None)
    if dtype is None:
        return torch.bfloat16
    if not isinstance(dtype, str):
        return dtype
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype.lower(), torch.bfloat16)


def main() -> None:
    parser = TrlParser((CustomScriptArguments, GRPOConfig, ModelConfig, SkillRetrievalArguments))
    script_args, training_args, model_args, skill_args = parser.parse_args_and_config()
    dataset_mode, dataset_subset_name, dataset = load_dataset_with_mode(script_args.dataset_name)
    reward_mode = script_args.reward_mode.strip().lower().replace("_", "-")
    if reward_mode != dataset_mode:
        raise ValueError(
            f"reward_mode ('{script_args.reward_mode}') must match dataset_mode ('{dataset_mode}'). "
            f"Please set --reward_mode {dataset_mode}."
        )

    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )
    if script_args.run_config:
        run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        skill_suffix = "_skill" if skill_args.use_skills else ""
        run_name = (
            f"GRPO{skill_suffix}_{model_name}_lr{lr_str}_bs{effective_batch_size}_"
            f"gen{training_args.num_generations}_temp{training_args.temperature}"
        )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=run_name,
            config={
                "method": "GRPO+Skill" if skill_args.use_skills else "GRPO",
                "dataset_name": script_args.dataset_name,
                "dataset_split": script_args.dataset_split,
                "dataset_mode": dataset_mode,
                "dataset_subset_name": dataset_subset_name,
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_generations": training_args.num_generations,
                "max_prompt_length": training_args.max_prompt_length,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "use_peft": model_args.use_peft,
                "loss_type": training_args.loss_type,
                "scale_rewards": training_args.scale_rewards,
                "reward_mode": reward_mode,
                "reward_binary_output": script_args.reward_binary_output,
                "skills_json_path": skill_args.skills_json_path if skill_args.use_skills else None,
                "skills_retrieval_mode": skill_args.skills_retrieval_mode if skill_args.use_skills else None,
                "skills_top_k": skill_args.skills_top_k if skill_args.use_skills else None,
            },
        )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=_resolve_model_dtype(model_args),
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = dataset[script_args.dataset_split]
    if skill_args.use_skills:
        format_prompt = make_grpo_raw_prompt_example(dataset_mode)
    else:
        format_prompt = make_grpo_format_prompt(tokenizer, dataset_mode)
    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    split_dataset = train_dataset.train_test_split(test_size=0.007, seed=42)

    trainer_cls = SkillAwareGRPOTrainer if skill_args.use_skills else GRPOTrainer
    trainer_kwargs = {
        "model": model_args.model_name_or_path,
        "reward_funcs": make_reward_function(
            mode=reward_mode,
            binary_output=script_args.reward_binary_output,
        ),
        "args": training_args,
        "train_dataset": split_dataset["train"],
        "eval_dataset": split_dataset["test"],
        "processing_class": tokenizer,
        "peft_config": get_peft_config(model_args),
    }
    if skill_args.use_skills:
        trainer_kwargs["skill_runtime"] = build_skill_runtime(skill_args)
    trainer = trainer_cls(**trainer_kwargs)

    resume_from_checkpoint = find_latest_checkpoint(training_args.output_dir)
    if resume_from_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
