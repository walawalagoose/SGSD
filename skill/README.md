# Math Skill Bank

This folder contains the procedures and components of skill bank.

## Format

A skill bank is a JSON object with the following top-level structure:

```json
{
  "general_skills": [],
  "task_specific_skills": {},
  "common_mistakes": [],
  "metadata": {}
}
```

SGSD retrieves ranked general skills and common mistakes, pairs them by rank, and uses each pair to construct one teacher. OPSD+Skill and GRPO+Skill inject the retrieved skill block into the training prompt.

## Bundled Artifacts

The repository includes complete cold-start artifacts for DAPO-Math:

- `skill/artifacts/dapo_math/qwen3_1b/`
- `skill/artifacts/dapo_math/qwen3_4b/`
- `skill/artifacts/dapo_math/qwen3_8b/`

Each directory contains generated memories, raw skill candidates, and a merged `claude_style_skills.json` skill bank. Additional historical artifacts are preserved under `skill/artifacts/dapo_math/`.

## Cold Start

Run the three-stage pipeline from the repository root. The default extraction and merge backend is local vLLM.

```bash
python -m skill.generate_math_memories \
  --dataset_name BytedTsinghua-SIA/DAPO-Math-17k \
  --dataset_split train \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --output_path outputs/skill_bank/generated_memories.json \
  --generation_backend vllm \
  --max_samples 256

python -m skill.generate_math_skills \
  --memory_path outputs/skill_bank/generated_memories.json \
  --output_path outputs/skill_bank/raw_skill_candidates.json \
  --llm_backend vllm \
  --model Qwen/Qwen3-4B

python -m skill.merge_math_skills \
  --skills_path outputs/skill_bank/raw_skill_candidates.json \
  --output_path outputs/skill_bank/claude_style_skills.json \
  --llm_backend vllm \
  --model Qwen/Qwen3-4B
```

`generate_math_memories.py` supports `vllm` and `transformers`. The extraction and merge tools support `vllm`, local `transformers`, and Azure backends.

## Retrieval

Training entries accept these common options:

```bash
--use_skills true \
--skills_json_path skill/artifacts/dapo_math/qwen3_1b/claude_style_skills.json \
--skills_retrieval_mode embedding \
--skills_embedding_model_path Qwen/Qwen3-Embedding-0.6B \
--skills_top_k 8
```

`embedding` and `random` retrieval are supported. `skills_top_k` controls the number of retrieved general skills and common mistakes.

## Online Writing

SGSD can update its skill bank during training:

```bash
--skills_enable_dynamic_update true \
--skills_update_frequency 25 \
--skills_update_threshold 0.8 \
--skills_max_new_skills 5 \
--skills_max_failures_to_analyze -1 \
--skills_dynamic_capacity 30 \
--skills_save_path outputs/sgsd/skill/updated_skills_latest.json
```

The updater extracts candidates from successful and failed trajectories, merges them within the current update window, and maintains bounded dynamic skill and mistake collections.
