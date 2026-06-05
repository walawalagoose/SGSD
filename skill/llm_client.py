from __future__ import annotations

import os
from typing import Protocol


class ChatClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...

    def batch_complete(self, prompts: list[str]) -> list[str]:
        ...


class AzureOpenAIChatClient:
    def __init__(
        self,
        *,
        model: str = "o3",
        api_key_env: str = "AZURE_OPENAI_API_KEY",
        endpoint_env: str = "AZURE_OPENAI_ENDPOINT",
        api_version_env: str = "AZURE_OPENAI_API_VERSION",
        default_api_version: str = "2025-01-01-preview",
        max_completion_tokens: int = 4096,
    ) -> None:
        from openai import AzureOpenAI

        api_key = os.environ.get(api_key_env)
        endpoint = os.environ.get(endpoint_env)
        api_version = os.environ.get(api_version_env, default_api_version)

        if not api_key or not endpoint:
            raise EnvironmentError(
                f"Azure OpenAI credentials are required. Please set {api_key_env} and {endpoint_env}."
            )

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.model = model
        self.max_completion_tokens = max_completion_tokens

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self.max_completion_tokens,
        )
        return response.choices[0].message.content or ""

    def batch_complete(self, prompts: list[str]) -> list[str]:
        return [self.complete(prompt) for prompt in prompts]


class LocalTransformersChatClient:
    def __init__(
        self,
        *,
        model: str,
        max_completion_tokens: int = 4096,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        temperature: float = 0.0,
        top_p: float = 1.0,
        trust_remote_code: bool = True,
    ) -> None:
        if not model or model == "o3":
            raise ValueError(
                "For the local skill backend, `model` must be a local checkpoint path or a Hugging Face model id."
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_completion_tokens = max_completion_tokens
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.do_sample = self.temperature > 0

        if dtype == "auto":
            torch_dtype = "auto"
        elif hasattr(torch, dtype):
            torch_dtype = getattr(torch, dtype)
        else:
            raise ValueError(f"Unsupported local dtype: {dtype!r}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, object] = {
            "trust_remote_code": trust_remote_code,
            "device_map": device_map,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)

    def _format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def _generate_one(self, prompt: str) -> str:
        import torch

        prompt_text = self._format_prompt(prompt)
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        input_device = next(self.model.parameters()).device
        encoded = {key: value.to(input_device) for key, value in encoded.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_completion_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = max(self.temperature, 1e-5)
            generation_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            generated = self.model.generate(**encoded, **generation_kwargs)

        completion_ids = generated[:, encoded["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()

    def complete(self, prompt: str) -> str:
        return self._generate_one(prompt)

    def batch_complete(self, prompts: list[str]) -> list[str]:
        return [self._generate_one(prompt) for prompt in prompts]


class LocalVLLMChatClient:
    def __init__(
        self,
        *,
        model: str,
        max_completion_tokens: int = 4096,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "auto",
        temperature: float = 0.0,
        top_p: float = 1.0,
        trust_remote_code: bool = True,
    ) -> None:
        if not model or model == "o3":
            raise ValueError(
                "For the local vLLM skill backend, `model` must be a local checkpoint path or a Hugging Face model id."
            )

        from transformers import AutoTokenizer
        from vllm import LLM

        self.max_completion_tokens = max_completion_tokens
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        llm_kwargs: dict[str, object] = {
            "model": model,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def _sampling_params(self):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_completion_tokens,
        )

    def complete(self, prompt: str) -> str:
        return self.batch_complete([prompt])[0]

    def batch_complete(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []

        formatted_prompts = [self._format_prompt(prompt) for prompt in prompts]
        outputs = self.llm.generate(formatted_prompts, self._sampling_params())
        responses: list[str] = []
        for output in outputs:
            if not output.outputs:
                responses.append("")
                continue
            responses.append(output.outputs[0].text.strip())
        return responses



def build_chat_client(
    *,
    backend: str = "azure",
    model: str = "o3",
    max_completion_tokens: int = 4096,
    local_device_map: str = "auto",
    local_dtype: str = "bfloat16",
    local_attn_implementation: str = "flash_attention_2",
    local_temperature: float = 0.0,
    local_top_p: float = 1.0,
    local_tensor_parallel_size: int = 1,
    local_gpu_memory_utilization: float = 0.9,
    local_max_model_len: int | None = None,
) -> ChatClient:
    backend = backend.strip().lower().replace("-", "_")
    if backend in {"azure", "azure_openai"}:
        return AzureOpenAIChatClient(model=model, max_completion_tokens=max_completion_tokens)
    if backend in {"local", "local_hf", "transformers"}:
        return LocalTransformersChatClient(
            model=model,
            max_completion_tokens=max_completion_tokens,
            device_map=local_device_map,
            dtype=local_dtype,
            attn_implementation=local_attn_implementation,
            temperature=local_temperature,
            top_p=local_top_p,
        )
    if backend in {"vllm", "local_vllm"}:
        return LocalVLLMChatClient(
            model=model,
            max_completion_tokens=max_completion_tokens,
            tensor_parallel_size=local_tensor_parallel_size,
            gpu_memory_utilization=local_gpu_memory_utilization,
            max_model_len=local_max_model_len,
            dtype=local_dtype,
            temperature=local_temperature,
            top_p=local_top_p,
        )
    raise ValueError(f"Unsupported LLM backend: {backend!r}")
