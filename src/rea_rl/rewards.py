"""Reward functions for GRPO training."""

import json
import math
import re
import torch
from typing import Dict

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from generation.math_verify.main import extract_math_answer, math_equiv
from pebble import ProcessPool
from collections import defaultdict
from generation.sft_data import full_regex


def processor(comp, gold):
    predict = comp[0]["content"]
    predict = extract_math_answer(predict)
    judge = math_equiv(gold, predict)
    return judge


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    with ProcessPool(max_workers=min(8, len(completions))) as pool:
        future = pool.map(processor, completions, solution, timeout=20)
        iterator = future.result()

        rewards = []
        for c in completions:
            try:
                rewards.append(next(iterator))
            except StopIteration:
                break
            except Exception as error:
                rewards.append(False)
                # print(error)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = kwargs["accuracy_reward"]

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def refine_len_reward(completions: list[Dict[str, str]], problem: list[str], accuracy_reward: list[bool], 
               completion_lengths: list[int], **kwargs) -> float:
    contents = [completion[0]["content"] for completion in completions]

    # Group lengths by problem
    problem_lengths = defaultdict(list)
    problem_min_max = {}
    for content, q, is_correct, completion_length in zip(contents, problem, accuracy_reward, completion_lengths):
        if is_correct:
            problem_lengths[q].append(completion_length)
    for q, lengths in problem_lengths.items():
        min_len = min(lengths)
        max_len = max(lengths)
        problem_min_max[q] = (min_len, max_len)

    rewards = []
    for content, q, is_correct, completion_length in zip(contents, problem, accuracy_reward, completion_lengths):
        if not is_correct:
            reward = -0.5
        else:
            min_len, max_len = problem_min_max[q]
            if max_len == min_len:
                reward = 0.0
            else:
                lambda_val = 0.5 - (completion_length - min_len) / (max_len - min_len)
                reward = lambda_val if is_correct else -0.5

        rewards.append(float(reward))

    return rewards


def reflection_reward(completions: list[Dict[str, str]], completion_ids: torch.Tensor, completion_lengths: list[int], threshold=225.09, **kwargs) -> float:
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, completion_id, completion_length in zip(contents, completion_ids, completion_lengths):
        # Extract content between <think> and </think> tags
        content_parts = content.split("<think>")[-1].split("</think>")[0]

        # Count "wait" occurrences (case-insensitive, consecutive count as one)
        wait_matches = re.findall(full_regex, content_parts)
        wait_count = len(wait_matches)

        # Assuming completion_id is a 1D tensor of token IDs
        think_start_id = 151648
        think_end_id = 151649
        completion_id = completion_id[: completion_length]
        start_indices = (completion_id == think_start_id).nonzero(as_tuple=True)[0]
        if len(start_indices) > 0:
            first_start_index = start_indices[0].item()
            completion_id = completion_id[first_start_index + 1 :]
        end_indices = (completion_id == think_end_id).nonzero(as_tuple=True)[0]
        if len(end_indices) > 0:
            first_end_index = end_indices[-1].item()
            completion_id = completion_id[: first_end_index]
        token_count = completion_id.size(0)

        # Calculate reward
        if token_count > 0:
            rewards.append(min(wait_count / token_count * threshold, 1))
        else:
            rewards.append(0.0)

    return rewards

def reflection_reward_60(completions: list[Dict[str, str]], completion_ids: torch.Tensor, completion_lengths: list[int], **kwargs) -> float:
    return reflection_reward(completions, completion_ids, completion_lengths, threshold=156.60)

def reflection_reward_90(completions: list[Dict[str, str]], completion_ids: torch.Tensor, completion_lengths: list[int], **kwargs) -> float:
    return reflection_reward(completions, completion_ids, completion_lengths, threshold=299.11)


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward
