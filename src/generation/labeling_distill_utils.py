import re
import json
import ijson
from copy import deepcopy
import argparse
from collections import defaultdict
from transformers import AutoTokenizer
from generation.utils import prompt_labeling

def extract_map(response, pattern=r'(\[\d+\]\.)'):
    parts = re.split(pattern, response)
    result = {}
    current_key = None
    for part in parts:
        if re.match(pattern, part):
            current_key = int(part.strip("\n").strip("*").strip(".").strip("[").strip("]"))  # Numerical part as key
        elif current_key:
            result[current_key] = part  # Subsequent part as value
            current_key = None  # Reset key
    return result


def extract_labels(response):
    # Split the input string by "[digit]." and filter out empty strings
    parts = extract_map("\n" + response, r'(\n\[\d+\]\.)')
    for k, part in parts.items():
        label_match = re.search(r"Label: (.+)", part)
        if not label_match:
            return {}
        parts[k] = label_match.group(1).strip()

    return parts


def main_parse(args):
    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_data = []
    for it in data:
        memory = []
        labels = extract_labels(it["_response"][0])
        sentences = it["generation"][it["cnt"][0]: it["cnt"][1]]
        if len(labels) != len(sentences):
            print(1)

        for cnt, sen in enumerate(sentences):
            if cnt + 1 in labels:
                memory.append((sen, cnt + it["cnt"][0], labels[cnt + 1].strip()))
        if len(all_data) and "".join(all_data[-1]["generation"]) == "".join(it["generation"]):
            all_data[-1]["_mem"] += deepcopy(memory)
        else:
            all_data.append(deepcopy(it))
            all_data[-1]["_mem"] = deepcopy(memory)
    print(len(all_data))

    prompts = []
    for it in all_data:
        if len(it["_mem"]) and it["_mem"][-1][1] == len(it["generation"]) - 1:
            it["_mem"] = it["_mem"][: -1]
        it["_mem"].append((it["generation"][-1], len(it["generation"]) - 1, "Right Result"))
        for sen, pos, label in it["_mem"]:
            if "right result" in label.lower():
                data_item = {
                    "problem": it["problem"],
                    "answer": it["_answer"],
                    "generation": it["generation"],
                    "pos": pos,
                    "_suffix": "\n\n".join(it["generation"][: pos + 1]) + "\n</think>\n\n**Final Answer:**\n\n\\boxed{"
                }
                prompts.append(deepcopy(data_item))
    print(len(prompts))

    output_path = args.file.replace(".json", "_generate.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, sort_keys=True, indent=4)

    prompts = []
    for it in all_data:
        if len(it["_mem"]) and it["_mem"][-1][1] == len(it["generation"]) - 1:
            it["_mem"] = it["_mem"][: -1]
        it["_mem"].append((it["generation"][-1], len(it["generation"]) - 1, "Right Result"))
        for sen, pos, label in it["_mem"]:
            if "right result" in label.lower():
                if args.use_asw:
                    data_item = {
                        "problem": it["problem"],
                        "_answer": it["_answer"],
                        "generation": it["generation"],
                        "pos": pos,
                        "_prompt": f'**Question**: {it["problem"]}\n\n**Gold Answer**: {it["_answer"]}\n\n**Response**: {it["generation"][pos]}\n\nEvaluate whether the model correctly answered the question. As long as the model provides the correct result, it counts as correct, regardless of format or wording. The response I provided is part of the complete response, so there\'s no need to include the entire reasoning process. Please judge only if the model has provided the correct answer up to this point. Please reason step by step first after "Reasoning:", then answer only with Yes or No after "Answer:".'
                    }
                else:
                    data_item = {
                        "problem": it["problem"],
                        "_answer": it["_answer"],
                        "generation": it["generation"],
                        "pos": pos,
                        "_prompt": f'**Question**: {it["problem"]}\n\n**Response**: {it["generation"][pos]}\n\nEvaluate whether the model have already answered the question. The response I provided is part of the complete response, so there\'s no need to include the entire reasoning process. Please judge only if the model has provided the answer up to this point. Please reason step by step first after "Reasoning:", then answer only with Yes or No after "Answer:".'
                    }
                prompts.append(deepcopy(data_item))
    print(len(prompts))

    output_path = args.file.replace(".json", "_parse.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, sort_keys=True, indent=4)


def main_clean(args):
    # Load the original JSON data from the specified file path
    input_path = args.file
    output_path = input_path.replace(".json", "_clean.json")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    filtered_data = defaultdict(list)
    with open(input_path, 'r') as f:
        data = ijson.items(f, 'item')
        for entry in data:
            if "cnt" in entry:
                entry["pos"] = entry["cnt"]
                entry.pop("cnt", None)
            if "Answer: Yes" not in entry["_response"][0]:
                entry["pos"] = None

            # Add the entry to the filtered list grouped by "question"
            prediction = "\n\n".join(entry["generation"])
            entry.pop("_response", None)
            entry.pop("_prompt", None)
            filtered_data[prediction].append(deepcopy(entry))

    # Combine the "pos" entries for the same question and only keep the first occurrence of other fields
    final_data = []
    for _, entries in filtered_data.items():
        combined_entry = entries[0].copy()  # Start with the first entry
        combined_entry["pos"] = sorted([entry["pos"] for entry in entries if entry["pos"] is not None])  # Combine all "pos" into a list
        final_data.append(deepcopy(combined_entry))
    print(len(final_data))

    all_prefix = all_length = 0
    for entry in final_data:
        if len(entry["pos"]) > 0:
            entry['_suffix'] = "\n\n".join(entry["generation"][: entry["pos"][0] + 1]) + "\n</think>\n\n**Final Answer:**\n\n"
        else:
            entry['_suffix'] = "\n\n".join(entry["generation"]) + "\n</think>\n\n**Final Answer:**\n\n"
        entry["answer"] = entry["_answer"]
        del entry["_answer"]
        if "_judge" in entry:
            entry["pre_judge"] = entry["_judge"]
            del entry["_judge"]
        post_token_len = len(tokenizer.encode(entry['_suffix']))
        full_token_len = len(tokenizer.encode("\n\n".join(entry["generation"])))
        all_prefix += post_token_len
        all_length += full_token_len

    # Save the modified data to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print(all_prefix, all_length, all_prefix / all_length)
    if args.train_file is None:
        return

    with open(args.train_file, 'w', encoding='utf-8') as f:
        for entry in final_data:
            if len(entry["pos"]):
                prompt = prompt_labeling(entry["problem"], entry["generation"], tokenizer, max_length=8192)
                answer = ["Think" for i, _ in enumerate(paragraphs)]
                for it in entry["pos"]:
                    answer[it] = "Result"
                answer = "\n\n".join([f"**[{i + 1}].** {p}" for i, p in enumerate(answer)])
                json.dump({
                    "text": answer,
                    "prefix": prompt
                }, f, ensure_ascii=False)
                f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument("--file", type=str, default=None, help="question and answer file")
    parser.add_argument("--train_file", type=str, default=None, help="output training file")
    parser.add_argument('--model_dir', type=str, default="Qwen2.5-32B-Instruct")
    parser.add_argument('--use_asw', action='store_true', default=False)
    args = parser.parse_args()

    if args.clean:
        main_clean(args)
    else:
        main_parse(args)
