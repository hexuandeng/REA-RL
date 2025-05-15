import os
import time
import json
import torch
import random
import multiprocessing
from openai import OpenAI
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer, GenerationConfig
from multiprocessing import Process, Queue, current_process, set_start_method

try:
    set_start_method('spawn')
except:
    pass

class GPUWorker(Process):
    def __init__(self, args, gpu_id, task_queue, result_queue, sampling_params):
        super().__init__()
        self.args = args
        self.gpu_id = gpu_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.llm = None
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        self.tokenizer.truncation_side = 'right'
        self.sampling_params = sampling_params

        stop_words = [self.tokenizer.eos_token, self.tokenizer.pad_token]
        config = GenerationConfig.from_pretrained(args.model_dir, trust_remote_code=True)
        if config.eos_token_id is not None:
            if isinstance(config.eos_token_id, int):
                config.eos_token_id = [config.eos_token_id]
            for it in config.eos_token_id:
                stop_words.append(self.tokenizer.convert_ids_to_tokens(it))
        if config.pad_token_id is not None:
            if isinstance(config.pad_token_id, int):
                config.pad_token_id = [config.pad_token_id]
            for it in config.pad_token_id:
                stop_words.append(self.tokenizer.convert_ids_to_tokens(it))
        stop_words = [i for i in set(stop_words) if i is not None]

        self.stop_words = stop_words

    def run(self):
        gpu_str = ",".join(map(str, self.gpu_id))
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        print(f"Process {current_process().name} using GPU {gpu_str}")

        # Initialize LLM instance
        self.llm = LLM(
            model=self.args.model_dir,
            tensor_parallel_size=len(self.gpu_id) // self.args.pipeline_parallel_size,
            pipeline_parallel_size=self.args.pipeline_parallel_size,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=self.args.max_model_len + self.args.max_tokens,
            gpu_memory_utilization=self.args.llm_gpu_memory_utilization,
            # disable_custom_all_reduce=True,
            seed=self.args.seed if self.args.seed else 0,
            dtype="bfloat16",
            enable_lora=self.args.lora is not None,
            enable_prefix_caching=True
        )
        print(f"Process {current_process().name} initialized on GPU {self.gpu_id}.")

        while True:
            try:
                it, prompt, kwargs = self.task_queue.get()
                generate_prompt = [i["prompt"][0] for i in prompt]
            except:
                print(f"Process {current_process().name} received termination signal.")
                break
            try:
                responses = self.llm.generate(
                    generate_prompt,
                    self.sampling_params,
                    use_tqdm=False,
                    lora_request=LoRARequest("labeling", 1, self.args.lora) if self.args.lora is not None else None
                )
                responses = sorted(responses, key=lambda x: int(x.request_id))
                assert len(prompt) == len(responses)
                outputs = []
                for p, r in zip(prompt, responses):
                    assert p["prompt"][0] == r.prompt
                    p["response"] = [out.text for out in r.outputs]
                    outputs.append(r)
            except Exception as e:
                print(f"Process {current_process().name} exited unexpectedly.")
                print(e)
                self.task_queue.put((it, prompt, kwargs))
                torch.cuda.empty_cache()
                continue
            self.result_queue.put((it, outputs))


class MultiGPUInference:
    def __init__(self, args, sampling_params, available_gpus=None):
        """
        Initializes the multi-GPU inference class.

        Args:
            args: Object containing the following attributes:
                - model_name_or_path: Name or path of the model
                - tensor_parallel_size: Tensor parallel size (default 1)
                - pipeline_parallel_size: Pipeline parallel size (default 1)
                - trust_remote_code: Whether to trust remote code (default True)
                - enforce_eager: Whether to enforce eager execution (default True)
                - max_model_len: Maximum length of the model (default 4096)
        """
        self.args = args
        self.available_gpus = available_gpus if available_gpus is not None else os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if not self.available_gpus:
            raise ValueError("No available GPUs detected. Please ensure at least one GPU is present and CUDA drivers are properly installed.")

        self.num_gpus = len(self.available_gpus)
        print(f"Detected {self.num_gpus} available GPUs: {self.available_gpus}")

        # Create task queue and result queue
        self.task_queue = Queue()
        self.result_queue = Queue()

        # Initialize worker processes for each GPU
        self.workers = []
        for i in range(0, len(self.available_gpus), args.n_gpu):
            worker = GPUWorker(
                args=args,
                gpu_id=self.available_gpus[i:i+args.n_gpu],
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                sampling_params=sampling_params
            )
            worker.start()
            self.workers.append(worker)

    def generate(self, prompts, *args, **kwargs):
        """
        Distributes inference tasks to different GPUs and collects the outputs.

        Args:
            prompts: A list of prompts (strings).
            sampling_params: A SamplingParams instance containing inference parameters.

        Returns:
            outputs: A list of generated results, in the same order as the input prompts.
        """
        if not isinstance(prompts, list):
            raise ValueError("prompts should be a list of strings.")
        if len(prompts) == 0:
            return [], []

        prompts = [{"id": cnt, "prompt": [it], "is_finish": False} for cnt, it in enumerate(prompts)]
        batch_size = max(int(len(prompts) / self.num_gpus) + 1, 8)
        batches = [prompts[i: i+batch_size] for i in range(0, len(prompts), batch_size)]
        num_requests = len(batches)
        for i, prompt in enumerate(batches):
            self.task_queue.put((i, prompt, {}))
        responses = []
        for i in tqdm(range(num_requests)):
            responses.append(self.result_queue.get())
        print(f"Generation {i + 1} among {num_requests} Finished!")
        responses = sorted(responses)
        outputs = [i for it in responses for i in it[1]]

        return outputs

    def exit(self):
        for _ in range(0, len(self.available_gpus), self.args.n_gpu):
            self.task_queue.put(None)


class APIWorker(Process):
    def __init__(self, args, task_queue, result_queue):
        super().__init__()
        self.args = args
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        lst = [
            (OpenAI(api_key="sk-***", base_url="https://api.deepseek.com"), "deepseek-reasoner"),
            (OpenAI(api_key="sk-***", base_url="https://api.deepseek.com"), "deepseek-reasoner"),
        ]
        client, model = random.choice(lst)

        while True:
            try:
                it, prompt, kwargs = self.task_queue.get()
            except:
                print(f"Process {current_process().name} received termination signal.")
                break

            reply = None
            for _ in range(10):
                try:
                    chat = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model,
                        temperature=self.args.temperature,
                        top_p=self.args.top_p,
                        max_completion_tokens=self.args.max_tokens
                    )
                    reply = chat.choices[0].message.model_extra['reasoning_content'] + "\n</think>\n" + chat.choices[0].message.content
                    break
                except Exception as e:
                    print(e)
                    time.sleep(10)
            if reply is None:
                print(f"{model} failed after ten attempts!!!")
                self.task_queue.put((it, prompt, kwargs))
                continue
            self.result_queue.put((it, reply))


class MultiAPIInference:
    def __init__(self, args):
        self.args = args
        self.task_queue = Queue()
        self.result_queue = Queue()

        self.n_workers = 16
        self.workers = []
        for _ in range(self.n_workers):
            worker = APIWorker(
                args=args,
                task_queue=self.task_queue,
                result_queue=self.result_queue,
            )
            worker.start()
            self.workers.append(worker)

    def generate(self, prompts):
        if not isinstance(prompts, list):
            raise ValueError("prompts should be a list of strings.")
        if len(prompts) == 0:
            return [], []

        num_requests = len(prompts)
        for i, prompt in enumerate(prompts):
            self.task_queue.put((i, prompt, {}))
        responses = []
        for i in tqdm(range(num_requests)):
            responses.append(self.result_queue.get())
        print(f"Generation {i + 1} among {num_requests} Finished!")
        responses = sorted(responses)
        outputs = [it[1] for it in responses]

        return outputs

    def exit(self):
        for _ in range(self.n_workers):
            self.task_queue.put(None)
