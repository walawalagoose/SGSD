import re

from math_verify import parse, verify

from . import math_dapo


def extract_boxed_answer(text):
    """
    Extract the answer from \\boxed{} format.
    For thinking models, only searches after </think> to avoid picking up
    intermediate answers from the thinking block.
    Handles nested braces correctly (e.g. \\boxed{\\frac{1}{2}}).
    """
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>") :] if think_end != -1 else text

    idx = search_text.find(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(search_text) and depth > 0:
        if search_text[i] == "{":
            depth += 1
        elif search_text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return search_text[start : i - 1].strip()
    return None


def _preprocess_for_parse(answer):
    """Convert ratio notation a:b -> \\frac{a}{b} so math_verify can parse it."""
    if answer is None:
        return None
    ratio_match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", answer)
    if ratio_match:
        return rf"\frac{{{ratio_match.group(1)}}}{{{ratio_match.group(2)}}}"
    return answer


def _ensure_text(value):
    if value is None:
        return ""
    return str(value)


def _completion_to_text(value):
    if isinstance(value, list):
        texts = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, list):
                    texts.extend(part.get("text", "") for part in content if isinstance(part, dict))
                else:
                    texts.append(str(content))
            else:
                texts.append(str(item))
        return "\n".join(text for text in texts if text)
    if isinstance(value, dict):
        content = value.get("content", "")
        if isinstance(content, list):
            return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content)
    return _ensure_text(value)


def _extract_ground_truths(kwargs):
    ground_truths = kwargs.get("ground_truth")
    if ground_truths is None:
        raise ValueError("Reward function requires dataset column 'ground_truth'.")
    return [_ensure_text(item) for item in ground_truths]


def _binaryize(score, binary_output):
    score = float(score)
    if not binary_output:
        return score
    return 1.0 if score > 0 else 0.0


def _load_dataset_compute_score(mode):
    dataset_mode = mode.strip().lower().replace("_", "-")

    if dataset_mode == "dapo-math":
        return math_dapo.compute_score

    raise ValueError(
        f"Unknown reward mode: {mode}. Supported modes are ['openthought', 'dapo-math']."
    )


def _default_reward_core(completions, ground_truths):
    rewards = []
    for completion, ground_truth in zip(completions, ground_truths):
        completion = _completion_to_text(completion)
        ground_truth = _ensure_text(ground_truth)
        pred_answer = extract_boxed_answer(completion)

        reward = 0.0

        gold_parsed = parse(ground_truth)
        pred_parsed = parse(_preprocess_for_parse(pred_answer))
        if gold_parsed is not None and pred_parsed is not None:
            try:
                reward = 1.0 if verify(gold_parsed, pred_parsed) else 0.0
            except Exception:
                pass

        if reward == 0.0:
            pred_norm = re.sub(r"\s+", "", pred_answer or "").lower()
            gt_norm = re.sub(r"\s+", "", ground_truth or "").lower()
            if pred_norm and pred_norm == gt_norm:
                reward = 1.0

        rewards.append(reward)

    return rewards


def _default_reward_with_feedback(completion, ground_truth):
    completion = _completion_to_text(completion)
    ground_truth = _ensure_text(ground_truth)
    pred_answer = extract_boxed_answer(completion)

    reward = 0.0
    correct = False

    gold_parsed = parse(ground_truth)
    pred_parsed = parse(_preprocess_for_parse(pred_answer))
    if gold_parsed is not None and pred_parsed is not None:
        try:
            correct = verify(gold_parsed, pred_parsed)
            reward = 1.0 if correct else 0.0
        except Exception:
            correct = False

    if reward == 0.0:
        pred_norm = re.sub(r"\s+", "", pred_answer or "").lower()
        gt_norm = re.sub(r"\s+", "", ground_truth or "").lower()
        if pred_norm and pred_norm == gt_norm:
            reward = 1.0
            correct = True

    if correct:
        feedback = None
    elif pred_answer is None:
        feedback = "Your previous attempt did not provide a valid final answer inside \\boxed{}."
    else:
        feedback = "Your previous attempt ended with an incorrect boxed final answer."

    return {
        "reward": reward,
        "correct": correct,
        "feedback": feedback,
        "pred_answer": pred_answer,
    }


def make_reward_scorer(mode="openthought", binary_output=True):
    """
    Build a richer scorer for custom trainers that need both rewards and feedback.

    Returns a callable with signature:
      scorer(completions, ground_truths) -> list[dict]
    where each dict contains at least `reward` and optionally `feedback`.
    """
    dataset_mode = mode.strip().lower().replace("_", "-")

    if dataset_mode == "openthought":

        def scorer(completions, ground_truths):
            results = []
            for completion, ground_truth in zip(completions, ground_truths):
                item = _default_reward_with_feedback(completion, ground_truth)
                item["reward"] = _binaryize(item["reward"], binary_output=binary_output)
                results.append(item)
            return results

        return scorer

    compute_score = _load_dataset_compute_score(dataset_mode)

    def scorer(completions, ground_truths):
        results = []
        for completion, ground_truth in zip(completions, ground_truths):
            completion = _completion_to_text(completion)
            ground_truth = _ensure_text(ground_truth)

            raw_result = compute_score(completion, ground_truth)
            if isinstance(raw_result, dict):
                score = raw_result.get("score", 0.0)
                pred = raw_result.get("pred")
            else:
                score = raw_result
                pred = None

            reward = _binaryize(score, binary_output=binary_output)
            correct = reward > 0
            if correct:
                feedback = None
            elif pred:
                feedback = "Your previous attempt ended with an incorrect final answer."
            else:
                feedback = "Your previous attempt did not produce a verifiably correct final answer."

            results.append(
                {
                    "reward": reward,
                    "correct": correct,
                    "feedback": feedback,
                    "pred_answer": pred,
                }
            )

        return results

    return scorer


def make_reward_function(mode="openthought", binary_output=True):
    """
    Build reward function for TRL GRPOTrainer.

    Modes:
      - openthought: current binary correctness reward.
      - dapo-math: route to dapo math compute_score.
    """
    dataset_mode = mode.strip().lower().replace("_", "-")

    if dataset_mode == "openthought":

        def reward_fn(completions, **kwargs):
            ground_truths = _extract_ground_truths(kwargs)
            return _default_reward_core(completions, ground_truths)

        return reward_fn

    compute_score = _load_dataset_compute_score(dataset_mode)

    def reward_fn(completions, **kwargs):
        ground_truths = _extract_ground_truths(kwargs)

        rewards = []
        for completion, ground_truth in zip(completions, ground_truths):
            completion = _completion_to_text(completion)
            ground_truth = _ensure_text(ground_truth)

            result = compute_score(completion, ground_truth)
            if isinstance(result, dict):
                score = result.get("score", 0.0)
            else:
                score = result

            rewards.append(_binaryize(score, binary_output=binary_output))

        return rewards

    return reward_fn
