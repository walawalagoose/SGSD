# SGSD: Skill-Conditioned Gated Self-Distillation for LLM Reasoning

<p align="center">
  <a href="https://arxiv.org/abs/2605.28791"><img src="https://img.shields.io/badge/arXiv-2605.28791-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
</p>

This is the official implementation of **Skill-Conditioned Gated Self-Distillation for LLM Reasoning**.

<p align="center">
<img src="images/sgsd_overview.png" width="80%" alt="SGSD Overview">
</p>

## News

- **[06/05/2026]** The initial code of SGSD has been released!
- **[05/27/2026]** We release the SGSD paper on [arXiv](ttps://arxiv.org/abs/2605.28791)!

## Overview

SGSD uses an experience-derived skill bank as teacher-side privileged information for on-policy self-distillation. Retrieved skill-mistake pairs define a pool of teachers, and a gated objective validates their token-level supervision against verifier outcomes before updating the student.

<details>
  <summary><strong>Abstract</strong></summary>

  On-policy self-distillation (SD) improves LLM reasoning by using teacher-side privileged information (PI) to turn sparse verifier outcomes into dense token-level supervision. Existing methods usually assume trusted PI, such as reference answers or successful traces. We ask whether PI can instead come from an experience-derived skill bank, where retrieved skills are compact and reusable but may also be irrelevant or misleading. We propose **S**kill-Conditioned **G**ated **S**elf-**D**istillation (**SGSD**), which formulates skill-based SD as teacher hypothesis validation rather than unconditional imitation. SGSD retrieves skill-mistake pairs, constructs a multi-teacher pool, and lets all skill-conditioned teachers score the same plain-prompt student rollout. The verifier validates each teacher's polarity: supporting a success or suppressing a failure gives positive supervision, while the opposite stance is reversed. A robust gated objective then distills informative teacher-student disagreements while suppressing uncertain or extreme signals. Experiments on multiple mathematical reasoning benchmarks show that SGSD consistently improves over GRPO and remains competitive with answer-conditioned OPSD under a weaker PI assumption. For example, on Qwen3-1.7B, SGSD outperforms GRPO by 6.2% and OPSD by 1.7% on average on AIME24, AIME25, and HMMT25.

</details>

## Installation

```bash
conda env create -f environment.yml
conda activate sgsd
pip install flash-attn==2.8.3 --no-build-isolation
```

The examples use [Qwen3](https://huggingface.co/Qwen), [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k), TRL, Accelerate, and vLLM. Local model and dataset paths are also accepted.

## Quick Start

The paper configuration for Qwen3-1.7B is provided in:

```bash
bash scripts/run_sgsd.sh
```

The script uses the bundled skill bank at `skill/artifacts/dapo_math/qwen3_1b/claude_style_skills.json`. Override its defaults with environment variables such as `MODEL_ID`, `DATASET_ID`, `SKILLS_JSON_PATH`, `OUTPUT_DIR`, and `CUDA_VISIBLE_IDS`.

Baseline launch scripts:

```bash
bash scripts/run_opsd.sh
bash scripts/run_grpo.sh
```

To enable skill-injected variants, add `--use_skills true` to the original training scripts. 
<!-- These variants share the same training entries as their plain counterparts. -->

## Skill Banks

Bundled skill banks and the cold-start generation pipeline are documented in [skill/README.md](skill/README.md). 
The public dataset adapter directly supports `BytedTsinghua-SIA/DAPO-Math-17k` and remains compatible with the earlier string-field dataset format.

## Training

## Evaluation

Evaluate a base model or trained checkpoint with plain prompts:

```bash
bash eval/run_eval.sh
CHECKPOINT_DIR=outputs/sgsd/checkpoint-200 bash eval/run_eval.sh
```

To evaluate with skills injected at inference:

```bash
bash eval/run_eval_with_skill.sh
```

## Citation

If you find this work helpful, please consider citing:


```bibtex
@article{huang2026skill,
  title={Skill-Conditioned Gated Self-Distillation for LLM Reasoning},
  author={Huang, Jiazhen and Chen, Xiao and Luo, Xiao and Dai, Yong and Hu, Senkang and Zhao, Yuzhi},
  journal={arXiv preprint arXiv:2605.28791},
  year={2026}
}
```

## Acknowledgements

This code builds on [OPSD](https://github.com/siyan-zhao/OPSD), [SkillRL](https://github.com/aiming-lab/SkillRL), [TRL](https://github.com/huggingface/trl), and the broader open-source reasoning community.
Thanks for making reproducible research easier.

## License

Released under the [MIT License](LICENSE).
