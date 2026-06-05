#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
DATASET_ID="${DATASET_ID:-BytedTsinghua-SIA/DAPO-Math-17k}"
SKILLS_JSON_PATH="${SKILLS_JSON_PATH:-skill/artifacts/dapo_math/qwen3_1b/claude_style_skills.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/opsd_skill}"
CUDA_VISIBLE_IDS="${CUDA_VISIBLE_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12950}"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_IDS}" "${ACCELERATE_BIN}" launch \
  --config_file "${REPO_ROOT}/accelerate.yaml" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${REPO_ROOT}/opsd_train.py" \
  --model_name_or_path "${MODEL_ID}" \
  --dataset_name "${DATASET_ID}" \
  --dataset_split train \
  --output_dir "${OUTPUT_DIR}" \
  --run_config OPSD_Skill_Qwen3-1.7B \
  --learning_rate 5e-6 \
  --max_grad_norm 0.1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --max_steps 300 \
  --save_steps 25 \
  --logging_steps 2 \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --torch_dtype bfloat16 \
  --max_completion_length 1024 \
  --max_length 20000 \
  --beta 0 \
  --lmbda 1 \
  --fixed_teacher \
  --jsd_token_clip 0.05 \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.6 \
  --vllm_tensor_parallel_size 1 \
  --use_peft \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --temperature 1.1 \
  --top_p 0.95 \
  --top_k 20 \
  --use_skills true \
  --skills_json_path "${SKILLS_JSON_PATH}" \
  --skills_retrieval_mode embedding \
  --skills_embedding_model_path Qwen/Qwen3-Embedding-0.6B \
  --skills_top_k 3 \
  --wandb_project OPSD
