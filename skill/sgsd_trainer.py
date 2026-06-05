from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import broadcast_object_list, gather_object, is_peft_model
from reward import make_reward_scorer
from reward.reward_adapter import extract_boxed_answer
from transformers import GenerationConfig
from trl.extras.profiling import profiling_decorator
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.sft_trainer import SFTTrainer

from opsd_trainer import OPSDTrainer as BaseOPSDTrainer, empty_cache

from .training import SkillRuntime
from .skill_updater import CallbackChatClient


class BaseSGSDTrainer(BaseOPSDTrainer):
    def __init__(
        self,
        *args,
        skill_runtime: SkillRuntime | None = None,
        skill_reward_mode: str = "openthought",
        teacher_use_reference_solution: bool = False,
        **kwargs,
    ) -> None:
        self.skill_runtime = skill_runtime
        self.skill_reward_scorer = make_reward_scorer(mode=skill_reward_mode, binary_output=True)
        self.teacher_use_reference_solution = teacher_use_reference_solution
        super().__init__(*args, **kwargs)

    def _score_completion_texts(
        self,
        completion_texts: list[str],
        ground_truths: list[str],
    ) -> list[dict[str, Any]]:
        return self.skill_reward_scorer(completion_texts, ground_truths)

    def _record_metric(self, mode: str, key: str, value: float) -> None:
        self._metrics[mode][key].append(float(value))

    def _record_completion_length_metrics(self, mode: str, completion_ids: torch.Tensor) -> None:
        if completion_ids.numel() == 0:
            return

        device = completion_ids.device
        max_length = int(completion_ids.shape[1])
        lengths = torch.full(
            (completion_ids.shape[0],),
            max_length,
            dtype=torch.long,
            device=device,
        )

        eos_token_id = getattr(self.generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self.processing_class, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_ids = []
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_token_ids = [int(token_id) for token_id in eos_token_id if token_id is not None]
        else:
            eos_token_ids = [int(eos_token_id)]

        eos_mask = torch.zeros_like(completion_ids, dtype=torch.bool)
        for token_id in eos_token_ids:
            eos_mask |= completion_ids == token_id
        has_eos = eos_mask.any(dim=1)
        first_eos = eos_mask.to(dtype=torch.long).argmax(dim=1)
        lengths = torch.where(has_eos, first_eos + 1, lengths)

        pad_token_id = getattr(self.processing_class, "pad_token_id", None)
        if pad_token_id is not None and int(pad_token_id) not in eos_token_ids:
            pad_mask = completion_ids == int(pad_token_id)
            has_pad = pad_mask.any(dim=1)
            first_pad = pad_mask.to(dtype=torch.long).argmax(dim=1)
            lengths = torch.where(~has_eos & has_pad, first_pad, lengths)

        gathered_lengths = self.accelerator.gather(lengths)
        gathered_lengths_float = gathered_lengths.float()
        self._metrics[mode]["completions/mean_length"].append(gathered_lengths_float.mean().item())
        self._metrics[mode]["completions/min_length"].append(gathered_lengths_float.min().item())
        self._metrics[mode]["completions/max_length"].append(gathered_lengths_float.max().item())

        stop_token_ids = list(eos_token_ids)
        if pad_token_id is not None:
            stop_token_ids.append(int(pad_token_id))
        if stop_token_ids:
            last_token_is_stop = torch.zeros(completion_ids.shape[0], dtype=torch.bool, device=device)
            last_tokens = completion_ids[:, -1]
            for token_id in stop_token_ids:
                last_token_is_stop |= last_tokens == token_id
            is_truncated = ~last_token_is_stop
        else:
            is_truncated = torch.ones(completion_ids.shape[0], dtype=torch.bool, device=device)

        gathered_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(gathered_is_truncated.float().mean().item())

        terminated_lengths = gathered_lengths[~gathered_is_truncated]
        if len(terminated_lengths) == 0:
            terminated_lengths = torch.zeros(1, dtype=torch.long, device=gathered_lengths.device)
        terminated_lengths_float = terminated_lengths.float()
        self._metrics[mode]["completions/mean_terminated_length"].append(terminated_lengths_float.mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(terminated_lengths_float.min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(terminated_lengths_float.max().item())

    def _append_generation_logs(
        self,
        prompt_texts: list[str],
        completion_texts: list[str],
        *,
        source: str,
    ) -> None:
        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(gather_object(completion_texts))

        for prompt, completion in zip(prompt_texts, completion_texts, strict=True):
            self._generation_outputs_buffer.append(
                {
                    "step": self.state.global_step,
                    "source": source,
                    "prompt": prompt,
                    "completion": completion,
                }
            )

    def _maybe_save_generation_buffer(self) -> None:
        if (
            self.state.global_step > 0
            and self.state.global_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(self.state.global_step)

    def _maybe_update_skills_from_records(self, records: list[dict[str, Any]]) -> None:
        if self.skill_runtime is None or not records:
            return
        self.skill_runtime.maybe_update_from_records(
            records,
            step=int(self.state.global_step) + 1,
            output_dir=self.args.output_dir,
            accelerator=self.accelerator,
        )

    def _build_skill_records(
        self,
        completion_texts: list[str],
        reward_items: list[dict[str, Any]],
        problems: list[str],
        ground_truths: list[str],
    ) -> list[dict[str, Any]]:
        records = []
        for problem, completion, reward_item, ground_truth in zip(
            problems, completion_texts, reward_items, ground_truths, strict=True
        ):
            records.append(
                {
                    "task": problem,
                    "prompt": problem,
                    "completion": completion,
                    "reward": float(reward_item.get("reward", 0.0)),
                    "ground_truth": str(ground_truth),
                    "predicted_answer": reward_item.get("pred_answer") or extract_boxed_answer(completion),
                    "data_source": "math",
                }
            )
        return records

    def _build_teacher_problem_prompt_text(
        self,
        problem_context: str,
        *,
        reference_solution: str | None = None,
    ) -> str:
        if self.teacher_use_reference_solution and reference_solution:
            user_message = (
                f"{problem_context}\n\n"
                "Here is a reference solution to this problem:\n"
                f"=== Reference Solution Begin ===\n{reference_solution}\n=== Reference Solution End ===\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
        else:
            user_message = (
                f"{problem_context}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
        messages = [{"role": "user", "content": user_message}]
        return self.processing_class.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def _build_main_model_updater_client(self, model: nn.Module):
        updater = None if self.skill_runtime is None else self.skill_runtime.updater
        if updater is None:
            return None

        def batch_complete(prompts: list[str]) -> list[str]:
            if not prompts:
                return []

            formatted_prompts = [self._format_skill_update_prompt(prompt) for prompt in prompts]
            prompt_ids, prompt_attention_mask = self._tokenize_prompt_texts(formatted_prompts)
            prompt_ids = prompt_ids.to(self.accelerator.device)
            prompt_attention_mask = prompt_attention_mask.to(self.accelerator.device)
            generation_config = GenerationConfig(
                max_new_tokens=updater.max_completion_tokens,
                do_sample=updater.local_temperature > 0,
                temperature=max(updater.local_temperature, 1e-5),
                top_p=updater.local_top_p,
                use_cache=True,
                pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self.processing_class.eos_token_id,
            )

            if self.use_vllm:
                self._wake_vllm_if_needed()
                _, _, completion_texts = self._generate_from_prompt_batch_vllm(prompt_ids, generation_config)
                return completion_texts

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                _, _, completion_texts = self._generate_from_prompt_batch_transformers(
                    unwrapped_model,
                    prompt_ids,
                    prompt_attention_mask,
                    int(prompt_ids.shape[1]),
                    generation_config,
                )
            return completion_texts

        return CallbackChatClient(batch_complete)

    def _format_skill_update_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        chat_template = getattr(self.processing_class, "chat_template", None)
        if chat_template:
            return self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def _teacher_adapter_context(self, model: nn.Module):
        if self.use_ema_teacher:
            return self._ema_teacher_context(model)
        if self.use_periodic_teacher:
            return self._periodic_teacher_context(model)
        if self.fixed_teacher and is_peft_model(model):
            return self.accelerator.unwrap_model(model).disable_adapter()
        return nullcontext()

    def _tokenize_prompt_texts(
        self,
        prompt_texts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_no_pad = self.processing_class(
            prompt_texts,
            padding=False,
            truncation=True,
            max_length=self.args.max_length,
        )
        prompt_lengths = [len(ids) for ids in encoded_no_pad["input_ids"]]
        max_prompt_len = max(prompt_lengths)
        encoded = self.processing_class(
            prompt_texts,
            padding="max_length",
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def _zero_like_loss(self, model: nn.Module, inputs: dict[str, torch.Tensor | Any]) -> torch.Tensor:
        outputs = model(
            input_ids=inputs["student_prompts"],
            attention_mask=inputs["student_prompt_attention_mask"],
        )
        return outputs.logits[..., 0].sum() * 0.0

    def _generate_from_prompt_batch_transformers(
        self,
        model: nn.Module,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        prompt_length: int,
        generation_config,
    ) -> tuple[torch.Tensor, list[str], list[str]]:
        original_use_cache = model.config.use_cache
        original_gen_use_cache = generation_config.use_cache
        model.config.use_cache = True
        generation_config.use_cache = True

        try:
            generated_outputs = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                use_cache=True,
            )
            generated_ids = generated_outputs.sequences
        finally:
            model.config.use_cache = original_use_cache
            generation_config.use_cache = original_gen_use_cache

        prompt_texts = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=False)
        completion_ids = generated_ids[:, prompt_length:]
        completion_texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        return generated_ids, prompt_texts, completion_texts

    def _generate_from_prompt_batch_vllm(
        self,
        prompt_ids: torch.Tensor,
        generation_config,
    ) -> tuple[torch.Tensor, list[str], list[str]]:
        device = self.accelerator.device
        prompt_texts = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=False)
        prompts_text_for_vllm = prompt_texts
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [text.replace(self.processing_class.pad_token, "") for text in prompt_texts]

        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0
        repetition_penalty = self.args.repetition_penalty if hasattr(self.args, "repetition_penalty") else 1.0
        min_p = self.args.min_p if hasattr(self.args, "min_p") else 0.0
        presence_penalty = self.args.presence_penalty if hasattr(self.args, "presence_penalty") else 0.0
        max_completion_length = generation_config.max_new_tokens

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text_for_vllm)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,
                    repetition_penalty=repetition_penalty,
                    temperature=generation_config.temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    presence_penalty=presence_penalty,
                    guided_decoding_regex=self.vllm_guided_decoding_regex,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text_for_vllm),
                (self.accelerator.process_index + 1) * len(prompts_text_for_vllm),
            )
            completion_ids = completion_ids[process_slice]
        elif self.vllm_mode == "colocate":
            guided_decoding = None
            if self.vllm_guided_decoding_regex:
                from vllm import SamplingParams
                from vllm.sampling_params import GuidedDecodingParams

                guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.vllm_guided_decoding_regex)
                sampling_params = SamplingParams(
                    n=1,
                    repetition_penalty=repetition_penalty,
                    temperature=generation_config.temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    presence_penalty=presence_penalty,
                    guided_decoding=guided_decoding,
                )
            else:
                from vllm import SamplingParams

                sampling_params = SamplingParams(
                    n=1,
                    repetition_penalty=repetition_penalty,
                    temperature=generation_config.temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    presence_penalty=presence_penalty,
                )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group)
                all_prompts_text = [text for prompt_list in gathered_prompts for text in prompt_list]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        completion_id_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        padded_completion_ids = []
        pad_token_id = self.processing_class.pad_token_id
        for completion_tensor in completion_id_tensors:
            if len(completion_tensor) > max_completion_length:
                padded_completion_ids.append(completion_tensor[:max_completion_length])
            elif len(completion_tensor) < max_completion_length:
                padding_needed = max_completion_length - len(completion_tensor)
                padded_completion_ids.append(
                    torch.cat(
                        [
                            completion_tensor,
                            torch.full(
                                (padding_needed,),
                                pad_token_id,
                                device=device,
                                dtype=completion_tensor.dtype,
                            ),
                        ]
                    )
                )
            else:
                padded_completion_ids.append(completion_tensor)

        padded_completion_ids = torch.stack(padded_completion_ids)
        generated_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)
        completion_texts = self.processing_class.batch_decode(padded_completion_ids, skip_special_tokens=True)
        return generated_ids, prompt_texts, completion_texts

    def _generate_rollouts(
        self,
        model: nn.Module,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        prompt_length: int,
        *,
        source: str,
    ) -> tuple[torch.Tensor, list[str], list[str]]:
        if self.use_vllm:
            self._wake_vllm_if_needed()
            return self._generate_from_prompt_batch_vllm(prompt_ids, self.generation_config)

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            return self._generate_from_prompt_batch_transformers(
                unwrapped_model,
                prompt_ids,
                prompt_attention_mask,
                prompt_length,
                self.generation_config,
            )

    def _assemble_inputs_from_items(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        student_prompts = torch.stack([item["student_prompt_ids"] for item in items])
        student_prompt_attention_mask = torch.stack([item["student_prompt_attention_mask"] for item in items])
        teacher_prompts = torch.stack([item["teacher_prompt_ids"] for item in items])
        teacher_prompt_attention_mask = torch.stack([item["teacher_prompt_attention_mask"] for item in items])
        completion_ids = torch.stack([item["completion_ids"] for item in items])

        student_input_ids = torch.cat([student_prompts, completion_ids], dim=1)
        student_attention_mask = torch.cat(
            [
                student_prompt_attention_mask,
                (completion_ids != self.processing_class.pad_token_id).to(student_prompt_attention_mask.dtype),
            ],
            dim=1,
        )
        teacher_input_ids = torch.cat([teacher_prompts, completion_ids], dim=1)
        teacher_attention_mask = torch.cat(
            [
                teacher_prompt_attention_mask,
                (completion_ids != self.processing_class.pad_token_id).to(teacher_prompt_attention_mask.dtype),
            ],
            dim=1,
        )

        labels = student_input_ids.clone()
        for row_index, item in enumerate(items):
            labels[row_index, : item["student_prompt_actual_length"]] = -100
        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100

        return {
            "student_input_ids": student_input_ids,
            "student_attention_mask": student_attention_mask,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "student_prompt_length": student_prompts.shape[1],
            "teacher_prompt_length": teacher_prompts.shape[1],
            "labels": labels,
        }

    def _run_loss_step(
        self,
        model: nn.Module,
        items: list[dict[str, Any]],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        assembled_inputs = self._assemble_inputs_from_items(items)
        return SFTTrainer.training_step(self, model, assembled_inputs, num_items_in_batch)

    def _finalize_step(self, loss: torch.Tensor, *, on_policy: bool) -> torch.Tensor:
        loss_scalar = float(loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        step_equiv = 1.0 / ga
        if on_policy:
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += step_equiv
        else:
            self._off_policy_loss_total += loss_scalar
            self._off_policy_step_equiv += step_equiv
        self._maybe_save_generation_buffer()
        return loss


class SGSDTrainer(BaseSGSDTrainer):
    def __init__(
        self,
        *args,
        gate_tau: float = 1.0,
        local_support_top_k: int = -1,
        polarity_clip_delta: float = 3.0,
        polarity_confidence_threshold: float = 0.05,
        **kwargs,
    ) -> None:
        if gate_tau <= 0:
            raise ValueError("gate_tau must be positive.")
        if local_support_top_k == 0 or local_support_top_k < -1:
            raise ValueError("local_support_top_k must be -1 or a positive integer.")
        if polarity_clip_delta <= 0:
            raise ValueError("polarity_clip_delta must be positive.")
        if polarity_confidence_threshold < 0:
            raise ValueError("polarity_confidence_threshold must be non-negative.")
        self.gate_tau = float(gate_tau)
        self.local_support_top_k = int(local_support_top_k)
        self.polarity_clip_delta = float(polarity_clip_delta)
        self.polarity_confidence_threshold = float(polarity_confidence_threshold)
        super().__init__(*args, **kwargs)
        self._mask_token_ids = self._build_mask_token_ids()
        if self._mask_token_ids:
            self._mask_token_ids_tensor = torch.tensor(
                sorted(self._mask_token_ids),
                dtype=torch.long,
            )
        else:
            self._mask_token_ids_tensor = None

    def _build_mask_token_ids(self) -> set[int]:
        tokenizer = self.processing_class
        mask_ids: set[int] = {
            int(token_id)
            for token_id in getattr(tokenizer, "all_special_ids", [])
            if token_id is not None
        }

        added_vocab = getattr(tokenizer, "get_added_vocab", lambda: {})()
        for token_text, token_id in added_vocab.items():
            if token_text in {"<think>", "</think>"}:
                mask_ids.add(int(token_id))
            elif token_text.startswith("<|") and token_text.endswith("|>"):
                mask_ids.add(int(token_id))

        for token_text in ("<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            token_ids = tokenizer.encode(token_text, add_special_tokens=False)
            if len(token_ids) == 1:
                mask_ids.add(int(token_ids[0]))

        for newline_count in range(1, 9):
            token_ids = tokenizer.encode("\n" * newline_count, add_special_tokens=False)
            if len(token_ids) == 1:
                mask_ids.add(int(token_ids[0]))

        return mask_ids

    def _build_token_mask(
        self,
        sampled_token_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask
        if self._mask_token_ids_tensor is not None:
            mask_ids = self._mask_token_ids_tensor.to(sampled_token_ids.device)
            mask = mask * (~torch.isin(sampled_token_ids, mask_ids)).to(dtype=valid_mask.dtype)
        return mask

    def _clip_support_delta(self, support_delta: torch.Tensor) -> torch.Tensor:
        return support_delta.clamp(min=-self.polarity_clip_delta, max=self.polarity_clip_delta)

    def _support_sign_from_scores(self, support_scores: torch.Tensor) -> torch.Tensor:
        support_sign = torch.sign(support_scores)
        return torch.where(
            support_scores.abs() > self.polarity_confidence_threshold,
            support_sign,
            torch.zeros_like(support_sign),
        )

    def _assemble_inputs_from_items(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        assembled = super()._assemble_inputs_from_items(items)
        device = assembled["student_input_ids"].device
        assembled["teacher_weights"] = torch.tensor(
            [float(item["teacher_weight"]) for item in items],
            device=device,
            dtype=torch.float32,
        )
        assembled["reward_outcomes"] = torch.tensor(
            [float(item["reward_outcome"]) for item in items],
            device=device,
            dtype=torch.float32,
        )
        assembled["example_group_ids"] = torch.tensor(
            [int(item["example_group_id"]) for item in items],
            device=device,
            dtype=torch.long,
        )
        return assembled

    def _compute_local_support_matching_terms(
        self,
        student_logits: torch.Tensor,
        teacher_target_logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        support_top_k = min(int(self.local_support_top_k), int(teacher_target_logits.shape[-1]))
        if support_top_k <= 0:
            raise ValueError("local_support_top_k must be positive in local support matching mode.")

        topk_indices = torch.topk(teacher_target_logits, k=support_top_k, dim=-1).indices
        sampled_indices = sampled_token_ids.unsqueeze(-1)
        sampled_in_topk = (topk_indices == sampled_indices).any(dim=-1, keepdim=True)

        # Keep the sampled token available for polarity estimation even when it is outside top-k.
        support_indices = torch.cat([topk_indices, sampled_indices], dim=-1)
        support_mask = torch.ones_like(support_indices, dtype=torch.bool)
        support_mask[..., -1] = (~sampled_in_topk).squeeze(-1)

        student_support_logits = torch.gather(student_logits, dim=-1, index=support_indices).to(torch.float32)
        teacher_support_logits = torch.gather(teacher_target_logits, dim=-1, index=support_indices).to(torch.float32)

        neg_inf = torch.finfo(student_support_logits.dtype).min
        student_support_logits = student_support_logits.masked_fill(~support_mask, neg_inf)
        teacher_support_logits = teacher_support_logits.masked_fill(~support_mask, neg_inf)

        student_support_log_probs = student_support_logits - torch.logsumexp(
            student_support_logits,
            dim=-1,
            keepdim=True,
        )
        teacher_support_log_probs = teacher_support_logits - torch.logsumexp(
            teacher_support_logits,
            dim=-1,
            keepdim=True,
        )
        student_support_log_probs = student_support_log_probs.masked_fill(~support_mask, neg_inf)
        teacher_support_log_probs = teacher_support_log_probs.masked_fill(~support_mask, neg_inf)

        sampled_match_mask = support_mask & (support_indices == sampled_indices)
        student_local_log_probs_sampled = student_support_log_probs.masked_fill(~sampled_match_mask, 0.0).sum(dim=-1)
        teacher_local_log_probs_sampled = teacher_support_log_probs.masked_fill(~sampled_match_mask, 0.0).sum(dim=-1)

        teacher_support_probs = torch.exp(teacher_support_log_probs).masked_fill(~support_mask, 0.0)
        local_support_divergence = (
            teacher_support_probs.detach() * (teacher_support_log_probs.detach() - student_support_log_probs)
        ).sum(dim=-1)
        support_sizes = support_mask.sum(dim=-1).to(dtype=torch.float32)

        return (
            student_local_log_probs_sampled.to(dtype=student_logits.dtype),
            teacher_local_log_probs_sampled.to(dtype=student_logits.dtype),
            local_support_divergence.to(dtype=student_logits.dtype),
            support_sizes,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        mode = "train" if model.training else "eval"
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]
        valid_mask = (shifted_labels != -100).to(dtype=torch.float32)
        loss_mask = self._build_token_mask(sampled_token_ids, valid_mask)

        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        with torch.no_grad():
            student_entropy_sum, student_entropy_count = self._token_entropy_stats_from_log_probs(
                student_log_probs.detach(),
                shifted_labels,
            )
        student_global_log_probs_sampled = torch.gather(
            student_log_probs,
            dim=-1,
            index=sampled_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        use_local_support_matching = self.local_support_top_k > 0
        if use_local_support_matching:
            student_log_probs_sampled = None
        else:
            student_log_probs_sampled = student_global_log_probs_sampled
            del student_logits
        del student_log_probs, outputs_student
        empty_cache()

        with torch.no_grad(), self._teacher_adapter_context(model):
            outputs_teacher = model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
            )
            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]
            del outputs_teacher
            empty_cache()

        teacher_target_logits = teacher_logits
        teacher_log_probs = F.log_softmax(teacher_target_logits, dim=-1)
        teacher_entropy_sum, teacher_entropy_count = self._token_entropy_stats_from_log_probs(
            teacher_log_probs,
            shifted_labels,
        )
        teacher_global_log_probs_sampled = torch.gather(
            teacher_log_probs,
            dim=-1,
            index=sampled_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        if use_local_support_matching:
            teacher_log_probs_sampled = None
        else:
            teacher_log_probs_sampled = teacher_global_log_probs_sampled
        del teacher_log_probs
        empty_cache()

        if use_local_support_matching:
            (
                student_log_probs_sampled,
                teacher_log_probs_sampled,
                local_row_token_losses,
                local_support_sizes,
            ) = self._compute_local_support_matching_terms(
                student_logits=student_logits,
                teacher_target_logits=teacher_target_logits,
                sampled_token_ids=sampled_token_ids,
            )
            del student_logits
        else:
            local_row_token_losses = None
            local_support_sizes = None

        del teacher_logits, teacher_target_logits
        empty_cache()

        self._accumulate_entropy_stats(
            mode=mode,
            student_entropy_sum=student_entropy_sum,
            student_entropy_count=student_entropy_count,
            teacher_entropy_sum=teacher_entropy_sum,
            teacher_entropy_count=teacher_entropy_count,
        )

        delta = teacher_log_probs_sampled.detach() - student_log_probs_sampled
        token_counts = loss_mask.sum(dim=1).clamp(min=1.0)
        raw_support_scores = (delta * loss_mask).sum(dim=1) / token_counts
        clipped_support_delta = self._clip_support_delta(delta)
        support_scores = (clipped_support_delta * loss_mask).sum(dim=1) / token_counts
        support_sign = self._support_sign_from_scores(support_scores)
        outcome_sign = torch.where(
            inputs["reward_outcomes"] > 0,
            torch.ones_like(support_scores),
            -torch.ones_like(support_scores),
        )
        polarity = outcome_sign * support_sign

        if use_local_support_matching:
            gate_inputs = -(local_row_token_losses.square()) / (2.0 * self.gate_tau)
            token_losses = math.log(2.0) - F.softplus(gate_inputs)
            row_losses = (token_losses * loss_mask).sum(dim=1) / token_counts
        else:
            gate_inputs = -(delta.square()) / (2.0 * self.gate_tau)
            token_losses = math.log(2.0) - F.softplus(gate_inputs)
            row_losses = (token_losses * loss_mask).sum(dim=1) / token_counts
        weighted_row_losses = inputs["teacher_weights"] * polarity * row_losses

        example_group_ids = inputs["example_group_ids"]
        unique_group_ids = torch.unique(example_group_ids, sorted=True)
        example_losses = [weighted_row_losses[example_group_ids == group_id].sum() for group_id in unique_group_ids]
        if example_losses:
            loss = torch.stack(example_losses).mean()
        else:
            loss = student_log_probs_sampled.sum() * 0.0

        self._record_metric(mode, "sgsd_support_score", support_scores.mean().item())
        self._record_metric(mode, "sgsd_support_score_raw", raw_support_scores.mean().item())
        self._record_metric(mode, "sgsd_gate_row_loss", row_losses.mean().item())
        if local_support_sizes is not None:
            self._record_metric(mode, "sgsd_local_support_divergence", local_row_token_losses.mean().item())
            self._record_metric(mode, "sgsd_local_support_size", local_support_sizes.mean().item())
        total_valid = valid_mask.sum().clamp(min=1.0)
        self._record_metric(mode, "sgsd_token_mask_keep_fraction", (loss_mask.sum() / total_valid).item())
        clipped_token_fraction = (
            ((delta.abs() > self.polarity_clip_delta).to(loss_mask.dtype) * loss_mask).sum()
            / loss_mask.sum().clamp(min=1.0)
        )
        self._record_metric(mode, "sgsd_clipped_token_fraction", clipped_token_fraction.item())
        neutral_by_threshold = (support_scores.abs() <= self.polarity_confidence_threshold).float().mean()
        self._record_metric(mode, "sgsd_below_threshold_fraction", neutral_by_threshold.item())
        self._record_metric(mode, "sgsd_helpful_fraction", (polarity > 0).float().mean().item())
        self._record_metric(mode, "sgsd_harmful_fraction", (polarity < 0).float().mean().item())
        self._record_metric(mode, "sgsd_neutral_fraction", (polarity == 0).float().mean().item())

        if return_outputs:
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()
            minimal_output.loss = loss
            return loss, minimal_output
        return loss

    @profiling_decorator
    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        mode = "train" if model.training else "eval"
        batch_size = inputs["student_prompts"].shape[0]

        student_generated_ids, student_prompt_texts, student_completion_texts = self._generate_rollouts(
            model,
            inputs["student_prompts"],
            inputs["student_prompt_attention_mask"],
            int(inputs["student_prompt_length"]),
            source="student",
        )
        student_completion_ids = student_generated_ids[:, int(inputs["student_prompt_length"]) :]
        self._record_completion_length_metrics(mode, student_completion_ids)
        student_reward_items = self._score_completion_texts(student_completion_texts, inputs["ground_truths"])
        self._record_metric(
            mode,
            "student_rollout_reward",
            sum(float(item["reward"]) for item in student_reward_items) / max(1, len(student_reward_items)),
        )
        self._append_generation_logs(student_prompt_texts, student_completion_texts, source="student")

        student_records = self._build_skill_records(
            student_completion_texts,
            student_reward_items,
            inputs["problems"],
            inputs["ground_truths"],
        )
        shared_client = None
        if (
            self.skill_runtime is not None
            and self.skill_runtime.updater is not None
            and self.skill_runtime.updater.uses_shared_main_model()
        ):
            shared_client = self._build_main_model_updater_client(model)
        update_scope = (
            self.skill_runtime.online_update_scope(enabled=True, shared_client=shared_client)
            if self.skill_runtime is not None
            else nullcontext()
        )
        with update_scope:
            self._maybe_update_skills_from_records(student_records)

        teacher_prompt_texts = []
        teacher_specs = []
        for example_index in range(batch_size):
            problem_contexts = inputs["sgsd_teacher_problem_contexts"][example_index]
            teacher_weights = inputs["sgsd_teacher_weights"][example_index]
            for problem_context, teacher_weight in zip(problem_contexts, teacher_weights, strict=True):
                teacher_prompt_texts.append(
                    self._build_teacher_problem_prompt_text(
                        problem_context,
                        reference_solution=inputs["teacher_reference_solutions"][example_index],
                    )
                )
                teacher_specs.append(
                    {
                        "example_index": example_index,
                        "teacher_weight": float(teacher_weight),
                        "reward_outcome": float(student_reward_items[example_index]["reward"]),
                    }
                )

        self._record_metric(mode, "sgsd_num_teachers", len(teacher_specs) / max(1, batch_size))

        if not teacher_specs:
            loss = self._zero_like_loss(model, inputs)
            return self._finalize_step(loss, on_policy=True)

        teacher_prompt_ids, teacher_attention_mask = self._tokenize_prompt_texts(teacher_prompt_texts)
        device = self.accelerator.device
        selected_items = []
        for row_index, teacher_spec in enumerate(teacher_specs):
            example_index = teacher_spec["example_index"]
            selected_items.append(
                {
                    "student_prompt_ids": inputs["student_prompts"][example_index],
                    "student_prompt_attention_mask": inputs["student_prompt_attention_mask"][example_index],
                    "student_prompt_actual_length": int(inputs["student_prompt_lengths_per_example"][example_index]),
                    "teacher_prompt_ids": teacher_prompt_ids[row_index].to(device),
                    "teacher_prompt_attention_mask": teacher_attention_mask[row_index].to(device),
                    "completion_ids": student_completion_ids[example_index],
                    "problem": inputs["problems"][example_index],
                    "ground_truth": inputs["ground_truths"][example_index],
                    "teacher_weight": teacher_spec["teacher_weight"],
                    "reward_outcome": teacher_spec["reward_outcome"],
                    "example_group_id": example_index,
                }
            )

        loss = self._run_loss_step(model, selected_items, num_items_in_batch)
        return self._finalize_step(loss, on_policy=True)
