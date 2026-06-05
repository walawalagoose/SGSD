# Math Skill Bank

This folder contains the skill-bank system used by SGSD, OPSD+Skill, GRPO+Skill, and Base+Skill evaluation. The bank is not an external tool set. It is a structured collection of reusable natural-language reasoning principles and common mistakes distilled from previous math trajectories.

## What Is Here

| Path | Purpose |
| --- | --- |
| `artifacts/` | Bundled DAPO-Math skill-bank artifacts used by the released experiments. |
| `generate_math_memories.py` | Samples model trajectories on math data and converts them into structured memory records. |
| `generate_math_skills.py` | Extracts raw skill or mistake candidates from each memory record. |
| `merge_math_skills.py` | Hierarchically merges raw candidates into a compact, deduplicated skill bank. |
| `skill_memory.py` | Loads a bank, retrieves relevant items, and formats them for prompts. |
| `skill_updater.py` | Maintains dynamic `dyn_###` skill and mistake items during SGSD training. |
| `training.py` | Connects retrieval, SGSD teacher construction, and online update triggers to the trainers. |
| `prompts/` | Prompt templates for memory generation, skill extraction, merging, and online updates. |

## Data Objects

The skill system has three levels of artifacts.

| Object | File | Meaning |
| --- | --- | --- |
| Memory records | `generated_memories.json` | Raw model attempts converted into normalized success/failure records. |
| Raw candidates | `raw_skill_candidates.json` | Per-memory extracted candidate skills or mistakes before global deduplication. |
| Merged bank | `claude_style_skills.json` | Final bank consumed by SGSD, OPSD+Skill, GRPO+Skill, and Base+Skill evaluation. |

Memory records preserve trajectory-level evidence. Raw candidates preserve locality by extracting from one memory at a time. The merged bank is the compact public interface used during training and evaluation.

## Bank Schema

A skill bank is a JSON object with four top-level keys:

```json
{
  "general_skills": [
    {
      "skill_id": "gen_001",
      "title": "Use invariants before calculation",
      "principle": "Identify quantities that remain fixed before expanding algebraic expressions.",
      "when_to_apply": "Use when a problem contains transformations or repeated operations."
    }
  ],
  "task_specific_skills": {},
  "common_mistakes": [
    {
      "mistake_id": "err_001",
      "description": "Assuming a variable is positive without checking constraints.",
      "why_it_happens": "The derivation silently imports an unstated domain assumption.",
      "how_to_avoid": "Check sign and domain constraints before multiplying or taking roots."
    }
  ],
  "metadata": {}
}
```

The released math implementation uses `general_skills` and `common_mistakes`. `task_specific_skills` is kept for schema compatibility with SkillRL-style banks, but it is not used by the math trainers.

Static cold-start items use IDs such as `gen_001` and `err_001`. Online updates use `dyn_###` IDs and are stored in the same `general_skills` and `common_mistakes` lists.

## Bundled Artifacts

The repository includes complete cold-start artifacts for DAPO-Math:

| Directory | Contents |
| --- | --- |
| `skill/artifacts/dapo_math/qwen3_1b/` | Qwen3-1.7B memories, raw candidates, and merged bank. |
| `skill/artifacts/dapo_math/qwen3_4b/` | Qwen3-4B memories, raw candidates, and merged bank. |
| `skill/artifacts/dapo_math/qwen3_8b/` | Qwen3-8B memories, raw candidates, and merged bank. |

Additional historical artifacts are preserved under `skill/artifacts/dapo_math/backup/` and `skill/artifacts/dapo_math/qwen3_4b_ori/`. They are kept for traceability and are not the default paths in the launch scripts.

The paper artifacts contain 28/24/46 general skills and 98/15/8 common mistakes for Qwen3-1.7B, Qwen3-4B, and Qwen3-8B, respectively.

## Cold-Start Pipeline

Run the three-stage pipeline from the repository root.

### 1. Generate Memories

`generate_math_memories.py` samples completions from a base model, scores them with the math verifier, and writes structured memory records. Successful records keep a compact refined trajectory. Failed records keep the incorrect attempt, predicted answer, ground truth, and inferred failure cues.

```bash
python -m skill.generate_math_memories \
  --dataset_name BytedTsinghua-SIA/DAPO-Math-17k \
  --dataset_split train \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --output_path outputs/skill_bank/generated_memories.json \
  --generation_backend vllm \
  --max_samples 256
```

Important options:

| Option | Meaning |
| --- | --- |
| `--generation_backend` | `vllm` for fast batched generation, or `transformers` for local HF generation. |
| `--max_samples` | Number of dataset examples used to build the cold-start bank. |
| `--max_new_tokens` | Completion budget for trajectory generation. |
| `--temperature`, `--top_p` | Sampling parameters for memory generation. |

The public DAPO adapter extracts the user problem from `prompt` and the answer from `reward_model.ground_truth`. The older string-field schema is still accepted.

### 2. Extract Raw Candidates

`generate_math_skills.py` compresses each memory and sends one memory per LLM call. Successful memories produce `general_skills`; failed memories produce `common_mistakes`. Each memory contributes at most three normalized candidates.

```bash
python -m skill.generate_math_skills \
  --memory_path outputs/skill_bank/generated_memories.json \
  --output_path outputs/skill_bank/raw_skill_candidates.json \
  --llm_backend vllm \
  --model Qwen/Qwen3-4B
```

Supported backends are `vllm`, `local`, and `azure`. The raw candidate file is intentionally not the final bank: it is usually redundant because candidates are extracted independently from many similar trajectories.

### 3. Merge Candidates

`merge_math_skills.py` compacts the raw candidate pool into a final bank. It merges general skills and common mistakes separately.

```bash
python -m skill.merge_math_skills \
  --skills_path outputs/skill_bank/raw_skill_candidates.json \
  --output_path outputs/skill_bank/claude_style_skills.json \
  --llm_backend vllm \
  --model Qwen/Qwen3-4B
```

The merge is hierarchical:

1. Split candidates into groups of `--merge_group_size` items.
2. Ask the merge model to consolidate each group.
3. Use the merged outputs as the next layer.
4. Stop when a root group is reached or `--merge_stagnation_patience` non-shrinking layers occur.
5. Deduplicate exact normalized items and reassign stable `gen_###` and `err_###` IDs.

Useful options:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--merge_group_size` | 32 | Number of candidate items per merge group. |
| `--batch_size` | 8 | Number of merge prompts sent per batch. |
| `--merge_stagnation_patience` | 3 | Stop after this many non-shrinking layers. Use 0 to disable this safeguard. |
| `--max_new_tokens` | 4096 | Completion budget for each merge response. |

## Retrieval

Training and evaluation entries use the same retrieval arguments:

```bash
--use_skills true \
--skills_json_path skill/artifacts/dapo_math/qwen3_1b/claude_style_skills.json \
--skills_retrieval_mode embedding \
--skills_embedding_model_path Qwen/Qwen3-Embedding-0.6B \
--skills_top_k 8
```

Retrieval modes:

| Mode | Behavior |
| --- | --- |
| `embedding` | Encodes the problem and all bank items with the embedding model, then ranks by cosine similarity. |
| `random` | Uses deterministic seeded sampling; dynamic `dyn_###` items are prioritized before static items. |

`skills_top_k` retrieves that many general skills and that many common mistakes. SGSD pairs the two ranked lists by rank, so top-8 retrieval creates up to eight teacher contexts. When embedding scores are available, SGSD normalizes pair scores with softmax weights.

## How Methods Consume the Bank

| Method | Student prompt | Teacher or rollout prompt | Evaluation prompt |
| --- | --- | --- | --- |
| SGSD | Plain problem | One rank-aligned skill-mistake pair per teacher | Plain problem |
| OPSD+Skill | Skill block plus problem | Same skill-augmented prompt | Plain problem |
| GRPO+Skill | Skill block plus problem | Skill-augmented rollout prompt | Plain problem |
| Base+Skill | Not trained | Not trained | Skill block plus problem |

SGSD uses skills as teacher-side privileged information. The student rollout is sampled from the plain problem, and each skill-conditioned teacher scores that same rollout. This is why SGSD can train with skills while evaluating without skill injection.

OPSD+Skill and GRPO+Skill are simpler prompt-injection baselines. They expose the training rollout to retrieved skills, but their checkpoints are evaluated with plain prompts in the paper.

Base+Skill uses `eval/evaluate_math_with_skill.py` or `eval/run_eval_with_skill.sh` and injects retrieved skills only at inference.

## Online Maintenance

Online maintenance is currently used by SGSD. Enable it with:

```bash
--skills_enable_dynamic_update true \
--skills_update_frequency 25 \
--skills_update_threshold 0.8 \
--skills_max_new_skills 5 \
--skills_max_failures_to_analyze -1 \
--skills_dynamic_capacity 30 \
--skills_save_path outputs/sgsd/skill/updated_skills_latest.json
```

The update loop works as follows:

1. SGSD collects student rollout records and verifier rewards.
2. Every `skills_update_frequency` optimizer steps, all ranks gather their records.
3. If the observed success rate is below `skills_update_threshold`, the updater runs.
4. Successful records are summarized into candidate general skills.
5. Failed records are summarized into candidate common mistakes.
6. New candidates are merged among themselves.
7. The merged new candidates are merged with existing dynamic `dyn_###` items.
8. Static cold-start items are left untouched.
9. Dynamic items are pruned by `skills_max_new_skills` and `skills_dynamic_capacity`.
10. The latest bank is written to `skills_save_path`, and a step-specific snapshot is written next to it.

If `skills_update_model` is unset and the backend is `local` or `vllm`, SGSD can reuse the current training model as the update client. This avoids loading a separate updater model. If `skills_update_model` is set, the updater uses that model instead. Azure backends always use their configured API client.

## Maintaining a Bank

Use these practices when creating or updating banks:

- Build the cold-start bank only from training data. Do not include evaluation problems.
- Keep `generated_memories.json` and `raw_skill_candidates.json`; they are useful for auditing why a merged skill exists.
- Treat `claude_style_skills.json` as the file consumed by training scripts.
- Preserve static IDs when editing a bank manually; use `dyn_###` only for online-maintained items.
- After manual edits, keep the same top-level schema and rerun retrieval with a small example before launching long training.
- If the bank grows too large, merge candidates again rather than appending an unbounded list.

## Prompt Templates

The text prompts used by the pipeline live in `skill/prompts/`:

| Prompt | Used by |
| --- | --- |
| `memory_generation_prompt.txt` | Generates memory-producing reasoning attempts. |
| `success_skill_from_memory_prompt.txt` | Extracts general skills from successful memories. |
| `failure_skill_from_memory_prompt.txt` | Extracts common mistakes from failed memories. |
| `merge_general_skills_prompt.txt` | Merges general skill candidates. |
| `merge_common_mistakes_prompt.txt` | Merges common mistake candidates. |
| `skill_update_prompt.txt` | Reserved for update-oriented prompting utilities. |

When adapting SGSD to a new domain, start by changing the memory generation and extraction prompts, then keep the same extract-merge-retrieve-maintain lifecycle.
