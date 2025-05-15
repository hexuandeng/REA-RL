import re
import json
import math
import argparse
from transformers import AutoTokenizer
from collections import defaultdict
from copy import deepcopy

def record_think_and_result_probabilities(logprobs):
    all_results = []

    for group in logprobs:
        results = []
        for token_group in group:
            if not token_group:
                continue

            max_entry = max(token_group, key=lambda x: x[0])
            if "Think" not in max_entry[1] and "Result" not in max_entry[1]:
                continue

            think_total = 0.0
            result_total = 0.0
            for logprob, token in token_group:
                if "think" in token.lower():
                    think_total += math.exp(logprob)
                elif "result" in token.lower():
                    result_total += math.exp(logprob)
            results.append(result_total / (think_total + result_total))
        all_results.append(results)

    return all_results


def find_max_probability(logprobs_results, all_len=None):
    if all_len is None:
        all_len = len(logprobs_results)
    max_len = all_len - 1 - min(int((all_len + 10) * 0.05), 5)
    first_strong_position = None
    first_strong_prob = None
    first_result_position = None
    first_result_position_end = None
    second_result_position = None
    max_probability_before_first_result = -float('inf')
    max_probability_position = None

    # Iterate through logprobs_results
    for pos, prob in enumerate(logprobs_results):
        if prob >= 0.25:
            if first_strong_position is None or \
                (prob > first_strong_prob and first_result_position is None):
                first_strong_position = pos
                first_strong_prob = prob
        if pos < all_len * 0.5:
            continue
        if prob >= 0.5:
            if first_result_position is None:
                first_result_position = pos
                first_result_position_end = pos
                connect = True
            elif second_result_position is None:
                if not connect:
                    second_result_position = pos
                else:
                    first_result_position_end = pos
        else:
            connect = False

        if first_result_position is None and prob > max_probability_before_first_result:
            max_probability_before_first_result = prob
            max_probability_position = pos

    if first_result_position is None:
        first_result_position = max_len
        first_result_position_end = max_len
    if second_result_position is None:
        second_result_position = first_result_position_end
    if max_probability_position is None:
        max_probability_position = first_result_position
    if first_strong_position is None:
        first_strong_position = first_result_position

    first_strong_position = min(first_strong_position, max_len)
    first_result_position = min(first_result_position, max_len)
    second_result_position = min(second_result_position, max_len)

    return [first_strong_position, first_result_position, second_result_position]


def split_sentence_by_newline(s, target, tokenizer):
    """
    Optimized splitting function: tokenizes only once, quickly locates the optimal split point
    using the token index of the newline character.
    """
    # 1. Single tokenize and get offset_mapping (assuming tokenizer supports it)
    encoding = tokenizer(s, return_offsets_mapping=True)
    offset_mapping = encoding.get("offset_mapping")
    total_tokens = len(encoding.tokens()) - 1
    target *= total_tokens

    # 2. Find the index of newline characters in the token list
    newline_token_indices = []
    # Precisely match newline characters using offset_mapping
    for i, (start, end) in enumerate(offset_mapping):
        if '\n' in s[start:end]:
            newline_token_indices.append(i)
    if not newline_token_indices:
        return s, total_tokens

    # 3. Find the position closest to the target
    closest_idx = min(range(len(newline_token_indices)),
                        key=lambda x: abs(newline_token_indices[x] - target - 2))
    best_newline_idx = newline_token_indices[closest_idx]
    _, split_pos = offset_mapping[best_newline_idx]
    best_prefix = s[: split_pos].strip()

    return best_prefix, total_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qaf", type=str, default="output/Qwen2.5-7B-Instruct/aime2024_test_math_cot_labeling", help="question and answer file")
    parser.add_argument('--model_dir', type=str, default="DeepSeek-R1-Distill-Qwen-7B")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    finals = []
    all_lengths = defaultdict(int)
    all_data = defaultdict(list)
    with open(args.qaf + ".json", "r", encoding="utf-8") as f:
        datas = json.load(f)
    for data in datas:
        paragraph = [i.strip() for i in data["generation"]]
        parse = record_think_and_result_probabilities(data["_logprobs"])[0][: len(paragraph)]
        if len(parse) != len(paragraph):
            print(len(parse), len(paragraph), data["_response"])
        pos = find_max_probability(parse, len(paragraph))
        chunk = {
            "Weak": "\n\n".join(paragraph[: pos[2] + 1]),
            "Normal": "\n\n".join(paragraph[: pos[1] + 1]),
            "Strong": "\n\n".join(paragraph[: pos[0] + 1])
        }

        all_lengths["Base"] += len(tokenizer.tokenize("\n\n".join(paragraph)))
        for k, v in chunk.items():
            cnt_data = deepcopy(data)
            del cnt_data["_logprobs"]
            del cnt_data["_response"]
            cnt_data["_suffix"] = v + "\n</think>\n\n**Final Answers:**\n\n"
            all_data[k].append(cnt_data)
            all_lengths[k] += len(tokenizer.tokenize(v))

    for k, v in all_lengths.items():
        if k == "Base":
            continue
        ratio = all_lengths[k] / all_lengths["Base"]
        with open(f"{args.qaf}_{k}.json", "w", encoding="utf-8") as f:
            json.dump(all_data[k], f, ensure_ascii=False, sort_keys=True, indent=4)

        lengths = 0
        total_lengths = all_lengths["Base"]
        remain_lengths = all_lengths[k]
        for data in datas:
            paragraph = "\n\n".join([i.strip() for i in data["generation"]])
            chunk, length = split_sentence_by_newline(paragraph, remain_lengths / total_lengths, tokenizer)

            cnt_data = deepcopy(data)
            del cnt_data["_logprobs"]
            del cnt_data["_response"]
            del cnt_data["_prompt"]
            cnt_data["_suffix"] = chunk.strip() + "\n</think>\n\n**Final Answers:**\n\n"
            all_data[k + "_Rand"].append(cnt_data)
            l = len(tokenizer.tokenize(chunk.strip()))
            remain_lengths -= l
            total_lengths -= length
            lengths += l

        assert total_lengths == 0
        print(k, all_lengths[k] / all_lengths["Base"])
        print(k + "_Rand", lengths / all_lengths["Base"])
        with open(f'{args.qaf}_{k}_Rand.json', "w", encoding="utf-8") as f:
            json.dump(all_data[k + "_Rand"], f, ensure_ascii=False, sort_keys=True, indent=4)

    for ratio in [0.5, 0.6, 0.7, 0.8, 0.9]:
        lengths = 0
        total_lengths = all_lengths["Base"]
        remain_lengths = total_lengths * ratio
        for data in datas:
            paragraph = "\n\n".join([i.strip() for i in data["generation"]])
            chunk, length = split_sentence_by_newline(paragraph, remain_lengths / total_lengths, tokenizer)

            cnt_data = deepcopy(data)
            del cnt_data["_logprobs"]
            del cnt_data["_response"]
            del cnt_data["_prompt"]
            cnt_data["_suffix"] = chunk.strip() + "\n</think>\n\n**Final Answers:**\n\n"
            all_data[ratio].append(cnt_data)
            l = len(tokenizer.tokenize(chunk.strip()))
            remain_lengths -= l
            total_lengths -= length
            lengths += l

        assert total_lengths == 0
        print(str(ratio) + "_Rand", lengths / all_lengths["Base"])
        with open(f'{args.qaf}_{ratio}_Rand.json', "w", encoding="utf-8") as f:
            json.dump(all_data[ratio], f, ensure_ascii=False, sort_keys=True, indent=4)
