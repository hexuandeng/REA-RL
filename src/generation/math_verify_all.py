import os
import re
import json
import argparse
from tqdm import tqdm
from pebble import ProcessPool # Import TimeoutError for specific handling
from collections import defaultdict
# Assuming generation.math_verify.main exists and works as intended
from generation.math_verify.main import extract_math_answer, math_equiv, INVALID_ANS
from transformers import AutoTokenizer
from functools import partial
import math # Import math for ceil

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def processor(data, model_dir, max_length=16384):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, truncation_side="right")
    responses = data["_response"]
    if not isinstance(responses, list):
        assert isinstance(responses, str)
        responses = [responses]

    results = []
    for response in responses:
        if "_suffix" in data:
            response = data["_suffix"] + response
        tokenized = tokenizer.encode(response, truncation=True, max_length=max_length, add_special_tokens=False)
        response = tokenizer.decode(tokenized)
        think_part = response.split("</think>")[0]
        tokenized_think = tokenizer.encode(think_part, add_special_tokens=False)

        predict = extract_math_answer(response)
        results.append({
            "problem": data["problem"],
            "answer": data["_answer"],
            "predict": response,
            "judge": predict,
            "lengths": len(tokenized),
            "think_lengths": len(tokenized_think)
        })

    return results

def math_judge(data):
    try:
        correct = math_equiv(data["answer"], data["judge"])
    except Exception as e:
        correct = False
        print(f"Error during math_equiv for gold='{data['answer']}', predict='{data['judge']}': {e}")
    data["judge"] = (correct, data["judge"])

    return data
    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--input_file', type=str, default="output/DeepSeek-R1-Distill-Qwen-7B/DeepScaleR_math_cot.json")
    args.add_argument('--model_dir', type=str, default="DeepSeek-R1-Distill-Qwen-7B")
    args.add_argument('--judge_output_file', type=str, default=None)
    args.add_argument('--dataset_output_file', type=str, default=None)
    args.add_argument('--max_length', type=int, default=16384) # Add chunk size argument
    args.add_argument('--chunk_size', type=int, default=20000) # Add chunk size argument
    args.add_argument('--workers', type=int, default=8)      # Add workers argument
    args.add_argument('--timeout', type=int, default=60)     # Add timeout argument, increased default
    args = args.parse_args()

    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items.")

    # Initialize aggregators before the loop
    new_data = []
    problem_used = set()
    correct_problem = []
    mem = []
    total_items = len(data)
    processed_count = 0
    token_count = 0
    think_token_count = 0
    valid_count = 0
    correct_count = 0
    chunk_size = args.chunk_size
    num_chunks = math.ceil(total_items / chunk_size)

    print(f"Processing data in {num_chunks} chunks of size {chunk_size}...")

    # Loop through data in chunks
    for i in tqdm(range(num_chunks), desc="Processing chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_items)
        chunk = data[start_idx:end_idx]
        print(f"\nProcessing chunk {i+1}/{num_chunks} (items {start_idx}-{end_idx-1})...")

        all_datas = []
        with ProcessPool(max_workers=args.workers) as pool:
            task_processor = partial(processor, model_dir=args.model_dir, max_length=args.max_length)
            future = pool.map(task_processor, chunk, timeout=args.timeout)
            iterator = future.result()
            for it in tqdm(iterator):
                all_datas += it

        with ProcessPool(max_workers=args.workers) as pool:
            task_processor = partial(math_judge)
            future = pool.map(task_processor, all_datas, timeout=args.timeout)
            iterator = future.result()

            for idx, original_data_item in tqdm(enumerate(all_datas)):
                try:
                    result = next(iterator)
                except StopIteration:
                    raise IndexError
                except Exception as error:
                    result = original_data_item
                    result["judge"] = (False, result["judge"])
                    print(error)

                processed_count += 1 # Increment global processed count
                token_count += result["lengths"]
                think_token_count += result["think_lengths"]
                mem.append([result["lengths"], result["think_lengths"]])
                valid_count += int(result["judge"][-1] != INVALID_ANS)
                correct_count += int(result["judge"][0])
                new_data.append({
                    "problem": result["problem"],
                    "answer": result["answer"],
                    "_answer": result["answer"],
                    "_response": result["predict"],
                    "_judge": (result["judge"], result["lengths"]) # judge is already (predict, correct) tuple
                })
                if result["judge"][0] and result["problem"] not in problem_used:
                    problem_used.add(result["problem"])
                    correct_problem.append(({
                        "problem": result["problem"],
                        "solution": result["answer"], # Or maybe store the first correct response? Sticking to original logic.
                    }))

    print("\n--- Processing Complete ---")

    # Final calculations and output (outside the loop)
    results = {
        "name": f"{args.input_file}_{args.max_length}",
        "Token": token_count / processed_count,
        "Think_Token": think_token_count / processed_count,
        "Valid": correct_count / total_items,
        "Acc": correct_count / processed_count,
        "Length": mem # Contains [length, think_length] for each response
    }
    print(f"{args.input_file}_{args.max_length}")
    print(f'Average Token Count per Response: {token_count / processed_count:.2f}')
    print(f'Average Acc: {correct_count / processed_count:.2f}')
    print(f'Total Problems Processed: {processed_count}')
    print(f"Total entries in new_data (responses): {len(new_data)}")
    print(f"Total entries in correct_problem: {len(correct_problem)}")

    print("Saving verification results...")
    output_verify_file = "output/verify.json"
    os.makedirs(os.path.dirname(output_verify_file), exist_ok=True) # Ensure output dir exists
    with open(output_verify_file, "a", encoding="utf-8") as outfile:
        json.dump(results, outfile, ensure_ascii=False, sort_keys=True)
        outfile.write("\n") # Ensure line separation
    print(f"Verification results appended to {output_verify_file}")

    if args.judge_output_file is not None:
        print(f"Saving detailed judge results to {args.judge_output_file}...")
        os.makedirs(os.path.dirname(args.judge_output_file), exist_ok=True)
        with open(args.judge_output_file, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4) # Keep indent for readability
        print("Judge results saved.")

    if args.dataset_output_file is not None:
        print(f"Saving correctly solved problems to {args.dataset_output_file}...")
        os.makedirs(os.path.dirname(args.dataset_output_file), exist_ok=True)
        with open(args.dataset_output_file, "w", encoding="utf-8") as outfile:
            for obj in correct_problem:
                json.dump(obj, outfile, ensure_ascii=False) # Removed sort_keys for consistency if needed
                outfile.write("\n") # Ensure line separation (JSON Lines format)
        print("Correctly solved problems saved.")

    print("Script finished.")