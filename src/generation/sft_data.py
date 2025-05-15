import re
import sys
import json
import ijson
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict, Counter

# Define the keywords
keywords = ["wait", "alternatively", "check", "but"]
keyword_pattern = "(?:" + "|".join(re.escape(k) for k in keywords) + ")"
separator_pattern = r".{0,59}"
full_regex = rf"(?i)({keyword_pattern}(?:{separator_pattern}{keyword_pattern})+|{keyword_pattern})"

if __name__ == "__main__":
    # --- Configuration ---
    judge_file_path = "output/DeepSeek-R1-Distill-Qwen-7B/deepscaler_labeling_distill_judge.json"
    ori_cot_file_path = "output/DeepSeek-R1-Distill-Qwen-7B/deepscaler_math_cot.json"

    question_right = defaultdict(list)
    tokenizer = AutoTokenizer.from_pretrained("DeepSeek-R1-Distill-Qwen-7B")

    get_all = {}
    with open(ori_cot_file_path, "r", encoding="utf-8") as f_olds:
        mems_stream = ijson.items(f_olds, 'item')
        for data in tqdm(mems_stream):
            for it in data["_response"]:
                prefix = it.split("</think>")[0].replace("\n", "").replace("\t", "").replace(" ", "")
                get_all[prefix] = it

    # Open both files simultaneously
    with open(judge_file_path, "r", encoding="utf-8") as f_datas:
        datas_stream = ijson.items(f_datas, 'item')
        print("Starting stream processing...")
        processed_count = 0
        skipped_count = 0

        # Iterate through both streams concurrently using zip
        for data in tqdm(datas_stream):
            processed_count += 1
            if not data["judge"][0][0]:
                skipped_count += 1
                continue
            # Process the 'math_cot' data stream item if condition passed
            try:
                response = "".join(data["generation"]).replace("\n", "").replace("\t", "").replace(" ", "")
                value_to_append = data["_suffix"] + data["_response"][0]
                if "\n</think>\n\n**Final Answer:**" not in value_to_append:
                    skipped_count += 1
                    continue
                question_right[data["problem"]].append((value_to_append + tokenizer.eos_token, get_all[response] + tokenizer.eos_token))
            except (KeyError, IndexError, TypeError) as e:
                print(f"Warning: Error accessing math_cot data in record {processed_count} (judge condition was True): {e}. Skipping.", file=sys.stderr)
                continue # Skip this pair if structure is unexpected

    print(f"Stream processing finished.")
    print(f"Total records processed: {processed_count}")
    print(f"Unique 'response' keys collected: {len(question_right)}")
    
    with open(f"dataset/labeling/SFT.json", "w", encoding="utf-8") as f:
        for k, v in tqdm(question_right.items()):
            for it in v[: 4]:
                json.dump({
                    "text": it[0],
                    "prefix": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {k}"
                }, f, ensure_ascii=False, sort_keys=True)
                f.write("\n")  # Ensure line separation [1](@ref)

    with open(f"dataset/labeling/RPO.json", "w", encoding="utf-8") as f:
        for k, v in tqdm(question_right.items()):
            for it in v[: 4]:
                json.dump({
                    "prompt": tokenizer.apply_chat_template([{
                        "role": "user", 
                        "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {k}"
                        }], tokenize=False, add_generation_prompt=True),
                    "chosen": it[0],
                    "rejected": it[1]
                }, f, ensure_ascii=False, sort_keys=True)
                f.write("\n")  # Ensure line separation [1](@ref)
