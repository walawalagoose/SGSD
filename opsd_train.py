from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import wandb
from transformers import AutoTokenizer, GenerationConfig
from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig

from data.dataset_adapter import load_dataset_with_mode, make_opsd_format_example
from opsd_trainer import OPSDTrainer
from skill.data_collator import SkillAwareOPSDDataCollator
from skill.training import SkillRetrievalArguments, build_skill_runtime
from train_resume import find_latest_checkpoint


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    use_tinker_loss: bool = field(
        default=False,
        metadata={"help": "Use the memory-efficient on-policy reverse KL loss."},
    )
    dataset_name: str = field(
        default="BytedTsinghua-SIA/DAPO-Math-17k",
        metadata={"help": "Dataset name or path passed to datasets.load_dataset()."},
    )
    dataset_split: str = field(default="train", metadata={"help": "Dataset split used for training."})
    fixed_teacher: bool = field(
        default=False,
        metadata={"help": "Use the initial policy as a fixed teacher. Requires use_peft=True."},
    )
    run_config: str | None = field(
        default=None,
        metadata={"help": "Run name used for the output directory suffix and W&B run name."},
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Presence penalty used during rollout generation."},
    )
    reason_first: bool = field(
        default=False,
        metadata={"help": "Generate a teacher analysis of the reference solution before distillation."},
    )
    top_k_loss: int = field(
        default=0,
        metadata={"help": "Restrict OPSD JSD to the teacher top-k tokens. Use 0 for the full vocabulary."},
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={"help": "Clip each OPSD token JSD loss. Use 0 to disable clipping."},
    )
    use_ema_teacher: bool = field(
        default=False,
        metadata={"help": "Use an exponential moving average teacher."},
    )
    ema_decay: float = field(
        default=0.999,
        metadata={"help": "EMA decay factor used when use_ema_teacher=True."},
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
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig, SkillRetrievalArguments))
    script_args, training_args, model_args, skill_args = parser.parse_args_and_config()
    dataset_mode, dataset_subset_name, dataset = load_dataset_with_mode(script_args.dataset_name)

    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher=True requires use_peft=True.")

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
            f"OPSD{skill_suffix}_{model_name}_lr{lr_str}_bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=run_name,
            config={
                "method": "OPSD+Skill" if skill_args.use_skills else "OPSD",
                "dataset_name": script_args.dataset_name,
                "dataset_split": script_args.dataset_split,
                "dataset_mode": dataset_mode,
                "dataset_subset_name": dataset_subset_name,
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "max_completion_length": training_args.max_completion_length,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "fixed_teacher": script_args.fixed_teacher,
                "reason_first": script_args.reason_first,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "jsd_token_clip": script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
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
    training_args.presence_penalty = script_args.presence_penalty

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with training_args.main_process_first(desc="load and preprocess dataset"):
        train_dataset = dataset[script_args.dataset_split]
        train_dataset = train_dataset.map(
            make_opsd_format_example(dataset_mode),
            remove_columns=train_dataset.column_names,
        )

    skill_runtime = build_skill_runtime(skill_args)
    data_collator = None
    if skill_runtime is not None:
        data_collator = SkillAwareOPSDDataCollator(
            tokenizer=tokenizer,
            max_length=training_args.max_length,
            reason_first=script_args.reason_first,
            skill_runtime=skill_runtime,
        )

    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
    )
    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        trainer.add_callback(LogCompletionsCallback(trainer, generation_config, num_prompts=8))

    resume_from_checkpoint = find_latest_checkpoint(training_args.output_dir)
    if resume_from_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
