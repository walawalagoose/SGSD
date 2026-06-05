#!/usr/bin/env bash
set -euo pipefail

# Evaluate a base model with retrieved skills. Set CHECKPOINT_DIR to evaluate a
# checkpoint with skill-augmented prompts.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
SKILLS_JSON_PATH="${SKILLS_JSON_PATH:-skill/artifacts/dapo_math/qwen3_1b/claude_style_skills.json}"
SKILLS_RETRIEVAL_MODE="${SKILLS_RETRIEVAL_MODE:-embedding}"
SKILLS_TOP_K="${SKILLS_TOP_K:-8}"
SKILLS_EMBEDDING_MODEL_PATH="${SKILLS_EMBEDDING_MODEL_PATH:-Qwen/Qwen3-Embedding-0.6B}"

DATASETS=(${DATASETS:-aime24 aime25 hmmt25})
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:--1}"
MIN_P="${MIN_P:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-38912}"
VAL_N="${VAL_N:-12}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
CUDA_VISIBLE_IDS="${CUDA_VISIBLE_IDS:-0}"
GPU_COUNT="$(echo "${CUDA_VISIBLE_IDS}" | tr ',' ' ' | wc -w | xargs)"

cd "${REPO_ROOT}"

EXTRA_ARGS=(
  --enable_thinking
  --skills_json_path "${SKILLS_JSON_PATH}"
  --skills_retrieval_mode "${SKILLS_RETRIEVAL_MODE}"
  --skills_top_k "${SKILLS_TOP_K}"
  --skills_embedding_model_path "${SKILLS_EMBEDDING_MODEL_PATH}"
)
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  EXTRA_ARGS+=(--checkpoint_dir "${CHECKPOINT_DIR}")
fi

for DATASET in "${DATASETS[@]}"; do
  NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_IDS}" "${PYTHON_BIN}" \
    "${REPO_ROOT}/eval/evaluate_math_with_skill.py" \
    --base_model "${BASE_MODEL}" \
    --dataset "${DATASET}" \
    --val_n "${VAL_N}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --min_p "${MIN_P}" \
    --presence_penalty "${PRESENCE_PENALTY}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor_parallel_size "${GPU_COUNT}" \
    "${EXTRA_ARGS[@]}"
done
