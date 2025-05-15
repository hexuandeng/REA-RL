import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import json
import random
from copy import deepcopy
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def create_prompt(args):
    dataset_name = args.qaf
    tokenizer = None
    if not args.api:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    if args.split == "json":
        with open(f"{dataset_name}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for cnt, it in enumerate(data):
            data[cnt]["_prompt"], data[cnt]["_answer"] = formatting_prompt(args, [it["_prompt"]], [it["_answer"]], tokenizer)
            if "_suffix" in it:
                data[cnt]["_prompt"] += it["_suffix"]
    elif args.split == "math_cot":
        with open(f"{dataset_name}.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        data = math_cot(args, dataset, tokenizer)
    elif args.split == "labeling":
        with open(f"{dataset_name}.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        data = math_labeling(args, dataset, tokenizer)
    elif "labeling_distill" in args.split:
        with open(f"{dataset_name}.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        data = labeling_distill(args, dataset, tokenizer, use_asw="use_asw" in args.split)
    if args.max_data is not None and len(data) > args.max_data:
        data = data[: args.max_data]

    return data


def formatting_prompt(args, prompts, answers, tokenizer):
    if args.prompt_id < 0 or args.prompt_id >= len(prompts):
        args.prompt_id = random.randint(0, len(prompts) - 1)
    prompt = prompts[args.prompt_id]
    answer = answers[args.prompt_id]
    if not args.api:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

    return prompt, answer


def split_text_with_numbers(text, tokenizer, return_len=False, max_length=None):
    if max_length is not None:
        tokenized = tokenizer.encode(text, truncation=True, max_length=max_length, add_special_tokens=False)
        text = tokenizer.decode(tokenized)
    parts = [p for p in re.split(r'(\n{2,})', text)
            if p.strip() and not re.match(r'^\n{2,}$', p)]
    
    current_result_paragraphs = []
    current_paragraph_parts = []
    current_paragraph_token_count = 0
    lens = []

    for part in parts:
        part_tokens = tokenizer.tokenize(part)
        part_token_count = len(part_tokens)
        sp = False
        if current_paragraph_token_count + part_token_count >= 32 and \
           part.strip() and part.strip()[-1] in "!.;?":
            sp = True
            current_paragraph_parts.append(part)
            current_paragraph_token_count += part_token_count
            part = ''
            part_token_count = 0
        elif current_paragraph_token_count + part_token_count >= 128:
            sp = True
            if not current_paragraph_parts:
                current_paragraph_parts.append(part)
                current_paragraph_token_count += part_token_count
                part = ''
                part_token_count = 0

        if sp:
            paragraph = "\n\n".join(current_paragraph_parts).strip()
            if paragraph.strip():
                current_result_paragraphs.append(paragraph)
                lens.append(current_paragraph_token_count)
            current_paragraph_parts = []
            current_paragraph_token_count = 0

        if part.strip():
            current_paragraph_parts.append(part)
            current_paragraph_token_count += part_token_count
        
    if current_paragraph_parts:
        paragraph = "\n\n".join(current_paragraph_parts).strip()
        current_result_paragraphs.append(paragraph)
        lens.append(current_paragraph_token_count)
    
    if return_len:
        return current_result_paragraphs, lens
    return current_result_paragraphs


def math_cot(args, dataset, tokenizer):
    results = []
    tokenizer = AutoTokenizer.from_pretrained("DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
    for data in tqdm(dataset):
        if "question" in data:
            data['problem'] = data['question']
            del data['question']
        prompts = [
            f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {data['problem']}"
        ]
        answers = [data['answer']]
        data["_prompt"], data["_answer"] = formatting_prompt(args, prompts, answers, tokenizer)
        if "_suffix" in data:
            tokenized = tokenizer.encode(data["_suffix"], add_special_tokens=False)
            if len(tokenized) > args.max_model_len:
                print(len(tokenized))
                tokenized = tokenized[: args.max_model_len]
                data["_suffix"] = tokenizer.decode(tokenized)
            data["_prompt"] += data["_suffix"]
        results.append(data)

    return results


def split_text_list(text_list, lens, n=1024):
    result = []
    current_chunk = []
    all_len = 0

    for text, l in zip(text_list, lens):
        all_len += l
        if all_len <= n:
            current_chunk.append(text)
        else:
            result.append(current_chunk)
            current_chunk = [text]
            all_len = l

    if current_chunk:
        result.append(current_chunk)

    return result


def labeling_distill(args, dataset, tokenizer, use_asw=True):
    results = []
    for data in tqdm(dataset):
        if not isinstance(data["_response"], list):
            assert isinstance(data["_response"], str)
            data["_response"] = [data["_response"]]
        for response in data["_response"]:
            prediction = response.split("</think>")[0]
            data["generation"], lens = split_text_with_numbers(prediction, tokenizer, return_len=True, max_length=args.max_model_len)
            data["full_response"] = response
            text_lists = split_text_list(data["generation"], lens)
            cnt = 0
            for text_list in text_lists:
                data = deepcopy(data)
                data["cnt"] = (cnt, cnt + len(text_list))
                cnt += len(text_list)
                response = "\n\n".join([f"**[{i + 1}].** {p}" for i, p in enumerate(text_list)])
                if use_asw:
                    prompts = [
                        f"**Question:** {data['problem']}\n\n**Gold Answer:** {data['answer']}\n\n**Response:** {response}\n\nYou are provided with a math Question, a Gold Answer and a model-generated Response. The response is divided into {len(text_list)} parts. For each part, analyze it and classify it based on its relationship to the provided context. For each part, assign one of the following labels:\n\n    - Reasoning: The part represents the reasoning process that leads to the answer.\n    - Right Result: The part is the answer provided by the model, where the model may provide the answer in the middle of its response, and the answer align with the Gold Answer.\n    - Wrong Result: Same as Right Result, but the answer do not align with the Gold Answer.\nFor each of the {len(text_list)} parts, please reply in format:\n\n[1]. Think: [Explanation for label choice]\nLabel: Reasoning/Right Result/Wrong Result\n\n[2]. Think: [Explanation for label choice]\nLabel: Reasoning/Right Result/Wrong Result\n\n..."
                    ]
                else:
                    prompts = [
                        f"**Question:** {data['problem']}\n\n**Response:** {response}\n\nYou are provided with a math Question and a model-generated Response. The response is divided into {len(text_list)} parts. For each part, analyze it and classify it based on its relationship to the provided context. For each part, assign one of the following labels:\n\n    - Reasoning: The part represents the reasoning process that leads to the answer.\n    - Right Result: The part is the answer provided by the model, where the model may provide the answer in the middle of its response, and the answer align with the Gold Answer.\n    - Wrong Result: Same as Right Result, but the answer do not align with the Gold Answer.\nFor each of the {len(text_list)} parts, please reply in format:\n\n[1]. Think: [Explanation for label choice]\nLabel: Reasoning/Right Result/Wrong Result\n\n[2]. Think: [Explanation for label choice]\nLabel: Reasoning/Right Result/Wrong Result\n\n..."
                    ]
                    
                #     - Verificartion: The part is the validation of the answer given by the model.\nIf both Result and Verificartion are in the same paragraph, please label it as Result.
                answers = [data['answer']]
                data["_prompt"], data["_answer"] = formatting_prompt(args, prompts, answers, tokenizer)
                results.append(deepcopy(data))
            assert cnt == len(data["generation"])

    return results


def prompt_labeling(question, response, tokenizer, max_length=None):
    if isinstance(response, str):
        response = response.split("</think>")[0].rstrip()
        paragraphs = split_text_with_numbers(response, tokenizer, max_length=max_length)
    elif isinstance(response, list):
        paragraphs = response
    else:
        raise NotImplemented
    
    text = "\n\n".join([f"**[{i + 1}].** {p}" for i, p in enumerate(paragraphs)])
    prompt_text = (
        f"**Question:** {question}\n\n"
        f"**Response:** {text}\n\n"
        f"You are provided with a math Question and a model-generated Response. "
        f"The response is divided into {len(paragraphs)} parts. For each part, analyze it "
        f"and classify it based on its relationship to the provided context. For each part, "
        f"assign one of the following labels:\n\n"
        f"    - Think: The part represents the reasoning process that leads to the answer.\n"
        f"    - Result: The part is the answer provided by the model, where the model may "
        f"provide the answer in the middle of its response.\n\n"
        f"For each of the {len(paragraphs)} parts, please reply in format:\n\n"
        f"[1]. Think/Result\n\n"
        f"[2]. Think/Result\n\n..."
    )

    return paragraphs, prompt_text
    

def math_labeling(args, dataset, tokenizer):
    results = []
    for data in tqdm(dataset):
        responses = deepcopy(data["_response"])
        for response in responses:
            data = deepcopy(data)
            data["generation"], prompt = prompt_labeling(data['problem'], response, tokenizer, max_length=args.max_model_len)
            data["_prompt"], _ = formatting_prompt(args, [prompt], [""], tokenizer)
            data["_prompt"] += "[1]."
            results.append(data)
            
    return results


def calc_avg_advantage(filename):
    total = 0
    count = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            advantage = data['advantage']

            if advantage <= 0:
                total += advantage
                count += 1

    if count > 0:
        average = total / count
        print(f"Average negative advantage: {average:.4f} (total {count} data points)")
    else:
        print("No negative advantage data found")


def calc_aha_density(json_path, tokenizer_path="DeepSeek-R1-Distill-Qwen-7B", to_print=False):
    with open(json_path, 'r') as f:
        data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # --- Worker Function ---
    def process_response(response_text, worker_tokenizer):
        try:
            # Extract content between <think> tags
            content_parts = response_text.split("<think>")[-1].split("</think>")[0]
            wait_matches = re.findall(full_regex, content_parts)
            wait_count = len(wait_matches)

            # Tokenize the content_parts (text between <think> tags)
            tokens = worker_tokenizer.encode(content_parts)
            token_len = len(tokens) # This is length of tokens *within* <think> tags
            history_val = token_len / wait_count if wait_count > 0 else float(token_len) # Consistent with original

            return wait_count, token_len, history_val
        except Exception as e:
            print(f"Error processing response snippet: {e}. Snippet start: {response_text[:100]}...")
            return None # Return None on error

    # --- Variables for collecting results from threads ---
    aggregated_total_wait_count = 0
    aggregated_total_token_count = 0
    aggregated_num_responses = 0
    aggregated_history = []
    futures = [] # To store Future objects

    tasks_submitted = 0
    with ThreadPoolExecutor(max_workers=64) as executor:
        for item in tqdm(data):
            if isinstance(item["_response"], str):
                item["_response"] = [item["_response"]]
            for response_text_content in item["_response"]:
                # Pass tokenizer and current_full_regex to the worker
                futures.append(executor.submit(process_response, response_text_content, tokenizer))
                tasks_submitted += 1

    # --- Retrieve results and aggregate ---
    for future in futures:
        try:
            wait_count, token_len, history_val = future.result()
            aggregated_total_wait_count += wait_count
            aggregated_total_token_count += token_len
            aggregated_history.append(history_val)
            aggregated_num_responses += 1
        except Exception as e:
            print(f"A task in the thread pool failed: {e}")

    # --- Final Reporting (using aggregated local variables) ---
    def calculate_percentile(p_data, percentile_val): # Renamed to avoid conflict
        if not p_data:
            return None
        # Ensure data is sorted for percentile calculation
        sorted_p_data = sorted(p_data) # Sort a copy, or ensure aggregated_history is sorted before calling
        index = (len(sorted_p_data) - 1) * percentile_val / 100
        lower_index = int(index)
        upper_index = lower_index + 1
        fraction = index - lower_index

        if upper_index < len(sorted_p_data):
            val = sorted_p_data[lower_index] + (sorted_p_data[upper_index] - sorted_p_data[lower_index]) * fraction
        else:
            # Handles cases where index is the last element or list is too short
            val = sorted_p_data[lower_index] if len(sorted_p_data) > 0 else 0

        # The print inside calculate_percentile was part of the original, kept for consistency
        if to_print: # only print if to_print is True
            print(f"{percentile_val}th percentile tokens/wait: {val:.2f}")
        return val

    if aggregated_num_responses > 0:
        avg_tokens_per_wait = aggregated_total_token_count / aggregated_total_wait_count if aggregated_total_wait_count > 0 else float('inf')

        if to_print:
            print(f"\n--- Final Statistics ---")
            print(f"Processed {aggregated_num_responses} responses.")

            # Percentile calculations
            aggregated_history = sorted(aggregated_history)
            p20_val = calculate_percentile(aggregated_history, 20)
            p40_val = calculate_percentile(aggregated_history, 40)
            median_val = calculate_percentile(aggregated_history, 50)
            p60_val = calculate_percentile(aggregated_history, 60)
            p80_val = calculate_percentile(aggregated_history, 80)
            p90_val = calculate_percentile(aggregated_history, 90)

            print(f"Average tokens/wait: {avg_tokens_per_wait:.2f}")

            percentage_wait = (aggregated_total_wait_count / aggregated_total_token_count) * 100 if aggregated_total_token_count > 0 else 0
            average_tokens_per_response_overall = aggregated_total_token_count / aggregated_num_responses
            print(f"Percentage of 'wait' occurrences relative to total tokens (within <think> tags): {percentage_wait:.2f}%")
            print(f"Average number of tokens (within <think> tags) per response processed: {average_tokens_per_response_overall:.2f}")
            print(f"Total 'wait' occurrences (within <think> tags): {aggregated_total_wait_count}")
            print(f"Total tokens processed (within <think> tags): {aggregated_total_token_count}")
        return avg_tokens_per_wait
    else:
        if to_print:
            print("\nNo processable '_response' lists found or successfully processed in the JSON data.")
        return None
