from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import wandb
from transformers import AutoTokenizer
from trl import (
    ModelConfig,
    SFTConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from data.dataset_adapter import load_dataset_with_mode, make_opsd_format_example
from train_resume import find_latest_checkpoint

from .sgsd_data_collator import SGSDDataCollator
from .sgsd_trainer import SGSDTrainer
from .training import SkillArguments, build_skill_runtime


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    dataset_name: str = field(
        default="BytedTsinghua-SIA/DAPO-Math-17k",
        metadata={"help": "Dataset name or path passed to datasets.load_dataset()."},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for training."},
    )
    reward_mode: str = field(
        default="dapo-math",
        metadata={"help": "Reward mode. Supported modes: 'openthought' and 'dapo-math'."},
    )
    run_config: str | None = field(
        default=None,
        metadata={"help": "Run name used for the output directory suffix and W&B run name."},
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Presence penalty used during rollout generation."},
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={"help": "Use the initial policy as a fixed teacher. Requires use_peft=True."},
    )
    use_ema_teacher: bool = field(
        default=False,
        metadata={"help": "Use an exponential moving average teacher instead of the live policy."},
    )
    ema_decay: float = field(
        default=0.999,
        metadata={"help": "EMA decay factor used when use_ema_teacher=True."},
    )
    use_periodic_teacher: bool = field(
        default=False,
        metadata={"help": "Use a lagged teacher snapshot refreshed from the student."},
    )
    periodic_teacher_update_steps: int = field(
        default=50,
        metadata={"help": "Optimizer-step period for refreshing the periodic teacher snapshot."},
    )
    teacher_use_reference_solution: bool = field(
        default=False,
        metadata={"help": "Include the dataset reference solution in teacher prompts."},
    )


@dataclass
class SGSDMethodArguments:
    sgsd_gate_tau: float = field(
        default=1.0,
        metadata={"help": "Temperature used by the SGSD gated loss."},
    )
    sgsd_local_support_top_k: int = field(
        default=-1,
        metadata={"help": "Teacher top-k local support width. Use -1 for full-vocabulary normalization."},
    )
    sgsd_polarity_clip_delta: float = field(
        default=3.0,
        metadata={"help": "Absolute clip bound for token-level support gaps."},
    )
    sgsd_polarity_confidence_threshold: float = field(
        default=0.05,
        metadata={"help": "Support-score threshold below which polarity is neutral."},
    )


@dataclass
class SGSDConfig(SFTConfig):
    """Training configuration containing only the runtime controls used by SGSD."""

    model_init_kwargs: dict[str, Any] | None = field(default=None, init=False)
    chat_template_path: str | None = field(default=None, init=False)
    dataset_text_field: str = field(default="text", init=False)
    dataset_kwargs: dict[str, Any] | None = field(default=None, init=False)
    dataset_num_proc: int | None = field(default=None, init=False)
    eos_token: str | None = field(default=None, init=False)
    pad_token: str | None = field(default=None, init=False)
    shuffle_dataset: bool = field(default=False, init=False)
    packing: bool = field(default=False, init=False)
    packing_strategy: str = field(default="bfd", init=False)
    padding_free: bool = field(default=False, init=False)
    pad_to_multiple_of: int | None = field(default=None, init=False)
    eval_packing: bool | None = field(default=None, init=False)
    completion_only_loss: bool | None = field(default=None, init=False)
    assistant_only_loss: bool = field(default=False, init=False)
    loss_type: str = field(default="nll", init=False)

    temperature: float = field(default=0.9, metadata={"help": "Sampling temperature for student rollouts."})
    top_p: float = field(default=0.95, metadata={"help": "Top-p sampling threshold for student rollouts."})
    top_k: int = field(default=0, metadata={"help": "Top-k sampling width for student rollouts."})
    max_completion_length: int = field(default=128, metadata={"help": "Maximum student rollout length."})
    use_vllm: bool = field(default=False, metadata={"help": "Use vLLM for student rollout generation."})
    vllm_mode: str = field(default="server", metadata={"help": "vLLM mode: `server` or `colocate`."})
    vllm_server_host: str = field(default="0.0.0.0", metadata={"help": "vLLM server host."})
    vllm_server_port: int = field(default=8001, metadata={"help": "vLLM server port."})
    vllm_server_timeout: float = field(default=240.0, metadata={"help": "vLLM server connection timeout."})
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={"help": "GPU memory utilization for colocated vLLM."},
    )
    vllm_tensor_parallel_size: int = field(default=1, metadata={"help": "Tensor parallel size for colocated vLLM."})
    vllm_guided_decoding_regex: str | None = field(default=None, metadata={"help": "Optional vLLM decoding regex."})
    vllm_sync_frequency: int = field(default=1, metadata={"help": "Student-to-vLLM weight synchronization frequency."})
    vllm_enable_sleep_mode: bool = field(default=False, metadata={"help": "Enable vLLM sleep mode."})
    log_completions: bool = field(default=False, metadata={"help": "Log sampled prompt-completion pairs."})
    log_completions_steps: int = field(default=100, metadata={"help": "Completion logging frequency."})
    num_completions_to_print: int = field(default=5, metadata={"help": "Number of completions to print."})
    wandb_entity: str | None = field(default=None, metadata={"help": "W&B entity."})
    wandb_project: str | None = field(default=None, metadata={"help": "W&B project."})
    wandb_run_group: str | None = field(default=None, metadata={"help": "W&B run group."})
    wandb_log_unique_prompts: bool = field(default=True, metadata={"help": "Log only unique prompts to W&B."})
    callbacks: list[str] = field(default_factory=list, metadata={"help": "Trainer callbacks."})

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_length is not None and self.max_completion_length >= self.max_length:
            raise ValueError("max_completion_length must be smaller than max_length.")


def _normalize_reward_mode(mode: str) -> str:
    return mode.strip().lower().replace("_", "-")


def _build_default_run_name(
    training_args: SGSDConfig,
    model_args: ModelConfig,
    method_args: SGSDMethodArguments,
    *,
    teacher_use_reference_solution: bool,
) -> str:
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )
    model_name = model_args.model_name_or_path.split("/")[-1]
    name = f"SGSD_{model_name}_lr{lr_str}_bs{effective_batch_size}_tau{method_args.sgsd_gate_tau}"
    if method_args.sgsd_local_support_top_k > 0:
        name = f"{name}_lsm{method_args.sgsd_local_support_top_k}"
    if teacher_use_reference_solution:
        name = f"{name}_withref"
    return name


def _resolve_model_dtype(model_args: ModelConfig) -> torch.dtype:
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        dtype = model_args.dtype
    else:
        return torch.bfloat16

    if not isinstance(dtype, str):
        return dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(dtype.lower(), torch.bfloat16)


def main() -> None:
    parser = TrlParser((CustomScriptArguments, SGSDConfig, ModelConfig, SkillArguments, SGSDMethodArguments))
    script_args, training_args, model_args, skill_args, method_args = parser.parse_args_and_config()

    if not skill_args.use_skills:
        raise ValueError("SGSD requires --use_skills true.")

    dataset_mode, dataset_subset_name, dataset = load_dataset_with_mode(script_args.dataset_name)
    reward_mode = _normalize_reward_mode(script_args.reward_mode)
    if reward_mode != dataset_mode:
        raise ValueError(
            f"reward_mode ('{script_args.reward_mode}') must match dataset_mode ('{dataset_mode}'). "
            f"Please set --reward_mode {dataset_mode}."
        )

    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher=True requires use_peft=True.")
    teacher_strategy_count = sum(
        bool(flag)
        for flag in (
            script_args.fixed_teacher,
            script_args.use_ema_teacher,
            script_args.use_periodic_teacher,
        )
    )
    if teacher_strategy_count > 1:
        raise ValueError("fixed_teacher, use_ema_teacher, and use_periodic_teacher are mutually exclusive.")
    if script_args.use_periodic_teacher and script_args.periodic_teacher_update_steps <= 0:
        raise ValueError("periodic_teacher_update_steps must be a positive integer.")

    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )
    if script_args.run_config:
        full_run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        full_run_name = _build_default_run_name(
            training_args,
            model_args,
            method_args,
            teacher_use_reference_solution=script_args.teacher_use_reference_solution,
        )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_run_name,
            config={
                "method": "SGSD",
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
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "fixed_teacher": script_args.fixed_teacher,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
                "use_periodic_teacher": script_args.use_periodic_teacher,
                "periodic_teacher_update_steps": (
                    script_args.periodic_teacher_update_steps if script_args.use_periodic_teacher else None
                ),
                "teacher_use_reference_solution": script_args.teacher_use_reference_solution,
                "sgsd_gate_tau": method_args.sgsd_gate_tau,
                "sgsd_local_support_top_k": method_args.sgsd_local_support_top_k,
                "sgsd_polarity_clip_delta": method_args.sgsd_polarity_clip_delta,
                "sgsd_polarity_confidence_threshold": method_args.sgsd_polarity_confidence_threshold,
                "skills_json_path": skill_args.skills_json_path,
                "skills_retrieval_mode": skill_args.skills_retrieval_mode,
                "skills_top_k": skill_args.skills_top_k,
                "skills_enable_dynamic_update": skill_args.skills_enable_dynamic_update,
                "skills_update_threshold": skill_args.skills_update_threshold,
                "skills_update_frequency": skill_args.skills_update_frequency,
                "skills_max_new_skills": skill_args.skills_max_new_skills,
                "skills_max_failures_to_analyze": skill_args.skills_max_failures_to_analyze,
                "skills_dynamic_capacity": skill_args.skills_dynamic_capacity,
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
        format_example = make_opsd_format_example(dataset_mode)
        train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)

    skill_runtime = build_skill_runtime(skill_args)
    data_collator = SGSDDataCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        skill_runtime=skill_runtime,
    )
    trainer = SGSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        fixed_teacher=script_args.fixed_teacher,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        use_periodic_teacher=script_args.use_periodic_teacher,
        periodic_teacher_update_steps=script_args.periodic_teacher_update_steps,
        skill_runtime=skill_runtime,
        skill_reward_mode=dataset_mode,
        teacher_use_reference_solution=script_args.teacher_use_reference_solution,
        gate_tau=method_args.sgsd_gate_tau,
        local_support_top_k=method_args.sgsd_local_support_top_k,
        polarity_clip_delta=method_args.sgsd_polarity_clip_delta,
        polarity_confidence_threshold=method_args.sgsd_polarity_confidence_threshold,
    )
    resume_from_checkpoint = find_latest_checkpoint(training_args.output_dir)
    if resume_from_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
