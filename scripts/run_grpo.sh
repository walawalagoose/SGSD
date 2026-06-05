#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
DATASET_ID="${DATASET_ID:-BytedTsinghua-SIA/DAPO-Math-17k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo}"
CUDA_VISIBLE_IDS="${CUDA_VISIBLE_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-19346}"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_IDS}" "${ACCELERATE_BIN}" launch \
  --config_file "${REPO_ROOT}/accelerate.yaml" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${REPO_ROOT}/grpo_train.py" \
  --model_name_or_path "${MODEL_ID}" \
  --dataset_name "${DATASET_ID}" \
  --dataset_split train \
  --reward_mode dapo-math \
  --output_dir "${OUTPUT_DIR}" \
  --run_config GRPO_Qwen3-1.7B \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 2 \
  --num_iterations 2 \
  --max_steps 300 \
  --save_steps 50 \
  --logging_steps 10 \
  --gradient_checkpointing \
  --max_prompt_length 2048 \
  --max_completion_length 16000 \
  --num_generations 8 \
  --temperature 1.2 \
  --beta 0 \
  --loss_type grpo \
  --scale_rewards group \
  --use_vllm \
  --vllm_mode colocate \
  --use_peft \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --wandb_project GRPO
