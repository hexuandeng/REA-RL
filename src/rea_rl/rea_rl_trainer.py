# Copyright 2025 The HuggingFace Team. All rights reserved.

import os
import json
import time
import requests
from typing import Any, Callable, Optional, Sized, Union
import warnings

import torch
from torch import nn
from torch.utils.data import Sampler
from accelerate.state import PartialState

from transformers import Trainer, AutoTokenizer
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed

from trl import GRPOTrainer
from trl.trainer.utils import pad
from trl.trainer.grpo_trainer import RepeatRandomSampler
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from generation.utils import prompt_labeling
from generation.labeling_to_prefix import record_think_and_result_probabilities, find_max_probability
from concurrent.futures import ThreadPoolExecutor, Future

class REARLTrainer(GRPOTrainer):
    def __init__(self, model, reward_funcs, args, *_args, **_kwargs):
        super().__init__(model, reward_funcs, args, *_args, **_kwargs)
        self.custom_truncate = args.custom_truncate
        self.custom_loss = args.custom_loss
        self.LabelClient = requests.Session()
        self.labeling_tokenizer = AutoTokenizer.from_pretrained(args.labeling_tokenizer)
        self.vllm_state = PartialState()
        self.acc = [0, 0]
        if args.local_rank == 0:
            self.executor = ThreadPoolExecutor(max_workers=1)
            self._future = None

    def _get_train_sampler(self) -> Sampler:
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
            seed=self.args.seed,
        )

    def _get_labeling(self, all_inputs, completion_texts, idx=1):
        classification_prompts = []
        all_paragraphs = []
        for i, line in enumerate(all_inputs):
            paragraphs, prompt_text = prompt_labeling(line['problem'], completion_texts[i], self.processing_class)
            all_paragraphs.append(paragraphs)
            classification_prompts.append(prompt_text)

        while True:
            try:
                classification_prompts = [
                    self.labeling_tokenizer.apply_chat_template([
                    {"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True) + "[1]."
                    for prompt in classification_prompts
                ]
                response = self.LabelClient.post(
                    f"http://{self.args.labeling_host}:{self.args.labeling_port}/generate/",
                    json={
                        "prompts": classification_prompts,
                        "n": 1,
                        "temperature": 0,
                        "max_tokens": 10240,
                        "logprobs": 5
                    },
                )
                if response.status_code == 200:
                    logprobs = response.json()["logprobs"]
                    texts = response.json()["texts"]
                    break
                else:
                    pass
            except Exception as e:
                print(e)
                time.sleep(30)

        pruned_inputs = []
        pruned_prefix = []
        for cnt, (input, logprob, paragraph) in enumerate(zip(all_inputs, logprobs, all_paragraphs)):
            parse = record_think_and_result_probabilities(logprob)[0][: len(paragraph)]
            pos = find_max_probability(parse, all_len=len(paragraph))
            user = "\n\n".join(paragraph[: pos[idx] + 1])
            if pos[idx] + 1 == len(paragraph):
                user = "\n\n".join(user.split("\n\n")[: -1])

            prompt = self.processing_class.apply_chat_template(input["prompt"], tokenize=False, add_generation_prompt=True)
            pruned_inputs.append(prompt + user.strip() + "\n</think>\n\n**Final Answer:**\n\n")
            pruned_prefix.append(user.strip() + "\n</think>\n\n**Final Answer:**\n\n")

        return pruned_inputs, pruned_prefix

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            all_inputs = gather_object(inputs)
            if self.state.global_step % self.num_iterations == 0 and self._step % self.args.gradient_accumulation_steps == 0:
                if self.accelerator.is_main_process:
                    if self._future is not None:
                        inputs = self._future.result()
                    else:
                        inputs = self._generate_and_score_completions(all_inputs)
                    # First, have main process load weights if needed
                    if self.use_vllm and self.state.global_step != self._last_loaded_step:
                        self._move_model_to_vllm()
                        self._last_loaded_step = self.state.global_step
                    self._future = self.executor.submit(self._generate_and_score_completions, all_inputs)
                else:
                    inputs = None

                self.vllm_state.wait_for_everyone()
                inputs = broadcast_object_list([inputs], from_process=0)[0]
                inputs = {
                    key: torch.chunk(tensor, self.args.world_size, dim=0)[self.accelerator.process_index] if tensor is not None else None
                    for key, tensor in inputs.items()
                }
                for k, v in inputs.items():
                    inputs[k] = v.to(self.accelerator.device)

                prompt_completion_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
                attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)  # (B, P+C)
                logits_to_keep = inputs["completion_ids"].size(1)  # we only need to compute the logits for the completion tokens
                with torch.inference_mode():
                    # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
                    # computation here, and use per_token_logps.detach() instead.
                    if self.num_iterations > 1:
                        inputs["old_per_token_logps"] = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )
                    else:
                        inputs["old_per_token_logps"] = None

                    if self.beta == 0.0:
                        inputs["ref_per_token_logps"] = None
                    elif self.ref_model is not None:
                        inputs["ref_per_token_logps"] = self._get_per_token_logps(
                            self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                        )
                    else:
                        with self.accelerator.unwrap_model(self.model).disable_adapter():
                            inputs["ref_per_token_logps"] = self._get_per_token_logps(
                                self.model, prompt_completion_ids, attention_mask, logits_to_keep
                            )

                split_dict = {
                    key: torch.chunk(tensor, self.args.gradient_accumulation_steps, dim=0) if tensor is not None else None
                    for key, tensor in inputs.items()
                }
                self._buffered_inputs = [
                    {key: split_dict[key][i] if split_dict[key] is not None else None for key in split_dict}
                    for i in range(self.args.gradient_accumulation_steps)
                ]
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            
            if self.log_completions:
                for cnt, (prompt, completion, reward) in enumerate(zip(inputs["prompt_ids"], inputs["completion_ids"], inputs["advantages"])):
                    prompt_text = self.processing_class.decode(prompt, skip_special_tokens=True)
                    completion_text = self.processing_class.decode(completion, skip_special_tokens=True)
                    with open(os.path.join(self.args.output_dir, "training_data_final.jsonl"), "a", encoding="utf-8") as f:
                        data = {
                            "prompt": prompt_text,
                            "completion": completion_text,
                            "advantage": reward.tolist()
                        }
                        if self.custom_loss:
                            data["loss_mask"] = self.processing_class.decode(completion[inputs["loss_mask"][cnt].bool()], skip_special_tokens=True)
                            data["loss_reward"] = inputs["loss_reward"][cnt].item()
                        json.dump(data, f, ensure_ascii=False, sort_keys=True)
                        f.write("\n")
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
            prompt_completion_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
            attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)  # (B, P+C)
            logits_to_keep = inputs["completion_ids"].size(1)  # we only need to compute the logits for the completion tokens
            with torch.inference_mode():
                # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
                # computation here, and use per_token_logps.detach() instead.
                if self.num_iterations > 1:
                    inputs["old_per_token_logps"] = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else:
                    inputs["old_per_token_logps"] = None

                if self.beta == 0.0:
                    inputs["ref_per_token_logps"] = None
                elif self.ref_model is not None:
                    inputs["ref_per_token_logps"] = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        inputs["ref_per_token_logps"] = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )
        return inputs

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        max_chunk_size = 4
        chunked_input_ids = torch.split(input_ids, max_chunk_size, dim=0)
        chunked_attention_mask = torch.split(attention_mask, max_chunk_size, dim=0)

        all_log_probs = []
        for chunk_input_ids, chunk_attention_mask in zip(chunked_input_ids, chunked_attention_mask):
            chunk_log_probs = super()._get_per_token_logps(model, chunk_input_ids, chunk_attention_mask, logits_to_keep)
            all_log_probs.append(chunk_log_probs)

        return torch.cat(all_log_probs, dim=0)

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        num_repeat = 1
        if self.custom_truncate:
            num_repeat = 2
        prompts = [item for item in prompts for _ in range(num_repeat)]
        prompt_ids = torch.repeat_interleave(prompt_ids, num_repeat, dim=0)
        prompt_mask = torch.repeat_interleave(prompt_mask, num_repeat, dim=0)

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        all_inputs = inputs # gather_object(inputs)
        all_prompts_text = prompts_text # gather_object(prompts_text)
        assert self.accelerator.is_main_process

        # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
        # num_generations outputs for each one. This is faster than generating outputs for each duplicate
        # prompt individually.
        ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
        with profiling_context(self, "vLLM.generate"):
            completion_ids = self.vllm_client.generate(
                prompts=ordered_set_of_prompts,
                n=self.num_generations,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=-1 if self.top_k is None else self.top_k,
                min_p=0.0 if self.min_p is None else self.min_p,
                max_tokens=self.max_completion_length,
                guided_decoding_regex=self.guided_decoding_regex,
            )
            completion_texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        if self.custom_truncate:
            pruned_inputs, pruned_prefix = self._get_labeling(all_inputs, completion_texts, idx=1)
            with profiling_context(self, "vLLM.generate"):
                pruned_ids = self.vllm_client.generate(
                    prompts=pruned_inputs,
                    n=1,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=256,
                    guided_decoding_regex=self.guided_decoding_regex,
                )
                pruned_texts = self.processing_class.batch_decode(pruned_ids, skip_special_tokens=True)
            final_completion_ids = []
            for cnt in range(len(completion_ids)):
                text = pruned_prefix[cnt] + pruned_texts[cnt] + self.processing_class.eos_token
                final_completion_ids.append(completion_ids[cnt])
                final_completion_ids.append(self.processing_class(text, add_special_tokens=False)["input_ids"])
            completion_ids = final_completion_ids

        # Pad the completions, and concatenate them with the prompts
        completion_lengths = [len(tensor) for tensor in completion_ids]
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        input_unsqueezes = [item for item in inputs for _ in range(num_repeat)]
        mem = {}
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in input_unsqueezes] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, completion_ids=completion_ids, completion_lengths=completion_lengths, **reward_kwargs, **mem)
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    mem[reward_func_name] = output_reward_func

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )
        # print(rewards_per_func)

        # Apply weights to each reward function's output and sum
        if self.custom_truncate:
            prompt_ids, prompt_mask, completion_ids, completion_mask, rewards_per_func = \
                self.filter_failed_data(prompt_ids, prompt_mask, completion_ids, completion_mask, rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations * num_repeat).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations * num_repeat).std(dim=1)

        if self.custom_truncate:
            acc_grouped = rewards_per_func[:, 0].view(-1, self.num_generations * num_repeat)
            group_all_le1 = (acc_grouped == 0).all(dim=1)
            # mean_grouped_rewards += group_all_le1.float() * 0.8312
            # std_grouped_rewards[group_all_le1] = 1 - 1e-4
            std_grouped_rewards[group_all_le1] = float("inf")

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations * num_repeat, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations * num_repeat, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)
        # print(advantages)

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            self._total_train_tokens += attention_mask.sum().sum().item() # self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        completion_length = completion_mask.sum(1).float().mean().item() # self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        finals = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
        }
        if self.custom_truncate and self.custom_loss:
            finals["loss_mask"], finals["loss_reward"] = self.filter_think_impact_data(rewards_per_func[:, 0], completion_ids, completion_mask)
            if self.custom_loss == 2:
                finals["loss_reward"] *= torch.max(torch.abs(advantages))
            else:
                finals["loss_reward"] *= torch.mean(torch.abs(advantages))

        return finals

    def filter_think_impact_data(
        self,
        rewards_flat: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor
    ): # Return type changed
        """
        Filters prompt/completion pairs (Torch tensors) based on outcome changes
        after modification (truncation before </think>). Returns dictionaries
        containing stacked tensors padded to original input dimensions.

        Includes:
        1. Optional check: Asserts or warns if the token sequence before </think> in
        the original completion does not match the beginning of the modified one.
        2. Specialized Masking: The 'completion_mask' in the output dictionary is
        a specialized mask where only tokens between the common prefix and
        </think> are marked as 1 (others are 0).

        Args:
            rewards_flat: 1D tensor of rewards for each completion.
            completion_ids: Padded IDs for all completions (N, seq_len_completion).
            completion_mask: Masks for padded completions (N, seq_len_completion).
        """
        # --- Input Validation ---
        if not isinstance(rewards_flat, torch.Tensor):
            raise TypeError("Expected rewards_flat to be a torch.Tensor")
        if not rewards_flat.ndim == 1:
            raise ValueError(f"Expected rewards_flat to be 1D, but got shape {rewards_flat.shape}")

        num_total_responses = rewards_flat.shape[0]
        think_token_id = self.processing_class.vocab.get("</think>")
        loss_mask = torch.zeros_like(completion_mask)
        loss_reward = torch.zeros_like(rewards_flat)
        correct_threshold = 0.5

        num_responses_to_process = num_total_responses
        if num_total_responses % 2 != 0:
            warnings.warn(f"Number of responses ({num_total_responses}) is odd. The last response will be ignored.", stacklevel=2)
            num_responses_to_process = (num_total_responses // 2) * 2

        # --- Main Loop ---
        for i in range(0, num_responses_to_process, 2):
            original_idx = i
            modified_idx = i + 1

            # --- Get Rewards & Check Correctness ---
            reward_original = rewards_flat[original_idx].float()
            reward_modified = rewards_flat[modified_idx].float()
            is_original_correct = not torch.isnan(reward_original) and reward_original > correct_threshold
            is_modified_correct = not torch.isnan(reward_modified) and reward_modified > correct_threshold

            # --- Process Completions & Find Prefix ---
            original_completion_seq = completion_ids[original_idx]
            modified_completion_seq = completion_ids[modified_idx]
            original_completion_attn_mask = completion_mask[original_idx].bool()
            modified_completion_attn_mask = completion_mask[modified_idx].bool()
            
            actual_original_tokens = original_completion_seq[original_completion_attn_mask]
            actual_modified_tokens = modified_completion_seq[modified_completion_attn_mask]

            think_indices_in_actual = torch.where(actual_original_tokens == think_token_id)[0]
            if len(think_indices_in_actual) == 0:
                common_len = actual_original_tokens.size()[0]
                original_prefix_actual = actual_original_tokens
            else:
                common_len = think_indices_in_actual[0]
                original_prefix_actual = actual_original_tokens[: common_len]

            think_indices_in_modified = torch.where(actual_modified_tokens == think_token_id)[0]
            if len(think_indices_in_modified) == 0:
                modified_len = actual_modified_tokens.size()[0]
                modified_prefix_actual = actual_modified_tokens
            else:
                modified_len = think_indices_in_modified[0]
                modified_prefix_actual = actual_modified_tokens[: modified_len]

            # --- ** Assertion/Check ** ---
            if modified_len + 1 >= common_len:
                continue

            # --- ** Specialized Mask Creation ** ---
            loss_mask[original_idx][modified_len: ] = 1
            loss_mask[modified_idx][modified_len: ] = 1
            if is_modified_correct:
                loss_reward[original_idx] = -1
                loss_reward[modified_idx] = 1

        loss_mask = (loss_mask & completion_mask).to(completion_mask.dtype)

        return loss_mask.detach(), loss_reward.detach()

    def filter_failed_data(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        rewards_per_func: torch.Tensor
    ): # Return type changed
        rewards_flat = rewards_per_func[:, 0].clone()
        if not isinstance(rewards_flat, torch.Tensor):
            raise TypeError("Expected rewards_flat to be a torch.Tensor")
        if not rewards_flat.ndim == 1:
            raise ValueError(f"Expected rewards_flat to be 1D, but got shape {rewards_flat.shape}")

        num_total_responses = rewards_flat.shape[0]
        correct_threshold = 0.5

        num_responses_to_process = num_total_responses
        if num_total_responses % 2 != 0:
            warnings.warn(f"Number of responses ({num_total_responses}) is odd. The last response will be ignored.", stacklevel=2)
            num_responses_to_process = (num_total_responses // 2) * 2

        # --- Main Loop ---
        for i in range(0, num_responses_to_process, 2):
            original_idx = i
            modified_idx = i + 1

            # --- Get Rewards & Check Correctness ---
            reward_original = rewards_flat[original_idx].float()
            reward_modified = rewards_flat[modified_idx].float()
            is_original_correct = not torch.isnan(reward_original) and reward_original > correct_threshold
            is_modified_correct = not torch.isnan(reward_modified) and reward_modified > correct_threshold

            # --- Process Completions & Find Prefix ---
            if is_modified_correct or not is_original_correct:
                self.acc[0] += 1
            self.acc[-1] += 1
            if not is_modified_correct:
                prompt_ids[modified_idx] = prompt_ids[original_idx].clone()
                prompt_mask[modified_idx] = prompt_mask[original_idx].clone()
                completion_ids[modified_idx] = completion_ids[original_idx].clone()
                completion_mask[modified_idx] = completion_mask[original_idx].clone()
                rewards_per_func[modified_idx] = rewards_per_func[original_idx].clone()

        print("Truncation Accuracy:", self.acc[0] / self.acc[-1])
        return prompt_ids, prompt_mask, completion_ids, completion_mask, rewards_per_func

    @profiling_decorator
    def custom_compute_loss(self, model, inputs, custom_mask=None, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        if custom_mask is None:
            custom_mask = completion_mask
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        if "per_token_logps" in inputs:
            per_token_logps = inputs["per_token_logps"]
        else:
            per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        sums = completion_mask.sum()
        # if self.custom_length_loss:
        #     sums = MAX_TOKENS * completion_ids.size(0)
        loss = (per_token_loss * custom_mask).sum() / sums
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * custom_mask).sum() / sums
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * custom_mask).sum() / sums
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss, per_token_logps

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, per_token_logps = self.custom_compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        if self.custom_loss:
            inputs["advantages"] = inputs["loss_reward"]
            inputs["per_token_logps"] = per_token_logps
            loss2, _ = self.custom_compute_loss(model, inputs, custom_mask=inputs["loss_mask"], return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
            loss += loss2
        return loss
