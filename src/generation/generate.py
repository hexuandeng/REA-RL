import os
import sys
import json
import torch
import argparse
from trl import TrlParser
from vllm import SamplingParams
from generation.utils import create_prompt
from generation.vllm_worker import MultiGPUInference, MultiAPIInference
from dataclasses import dataclass, field
from typing import List, Optional, Literal

torch.set_num_threads(12)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class BaseConfig:
    # Model Config
    model_dir: Optional[str] = field(
        default=None, metadata={"help": "llm model dir"}
    )
    lora: Optional[str] = field(
        default=None, metadata={"help": "lora parameters, default is None"}
    )
    api: bool = field(
        default=False, metadata={"help": "flag to enable API mode"}
    )
    max_model_len: int = field(
        default=16384, metadata={"help": "maximum model length"}
    )
    llm_gpu_memory_utilization: float = field(
        default=0.97, metadata={"help": "gpu memory utilization for policy. rm will use 15G memory, so the remaining memory is used for llm."}
    )
    pipeline_parallel_size: int = field(
        default=1, metadata={"help": "llm tensor parallel size"}
    )
    n_gpu: int = field(
        default=1, metadata={"help": "num gpu per worker"}
    )

    # Sampling Config
    temperature: float = field(
        default=0, metadata={"help": "control diversity of llm generation"}
    )
    top_p: float = field(
        default=1.0, metadata={"help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens."}
    )
    top_k: int = field(
        default=-1, metadata={"help": "Float that controls the probability of other highly-scored candidates to be chosen"}
    )
    max_tokens: int = field(
        default=2048, metadata={"help": "Maximum number of tokens to generate per output sequence."}
    )
    n_generate_sample: int = field(
        default=1, metadata={"help": "how many samples generated for each step. B2 in paper."}
    )
    stop: Optional[List[str]] = field(
        default=None, metadata={"help": "possible stop tokens for each step"}
    )
    logprobs: Optional[int] = field(
        default=None, metadata={"help": "logprobs parameter, default is None"}
    )
    seed: Optional[int] = field(
        default=2, metadata={"help": "seed of llm generation for reproducible"}
    )

    # data config
    qaf: str = field(
        default=None, metadata={"help": "dataset path"}
    )
    split: str = field(
        default="math_eval", metadata={"help": "split parameter, default is 'math_eval'"}
    )
    prompt_id: int = field(
        default=0, metadata={"help": "prompt id, default is 0"}
    )
    max_data: Optional[int] = field(
        default=None, metadata={"help": "maximum data amount, default is None"}
    )
    save_suffix: str = field(
        default="", metadata={"help": "suffix of the saved file name, default is an empty string"}
    )


if __name__ == '__main__':
    parser = TrlParser(BaseConfig)
    config = parser.parse_args_and_config()[0]
    print(config)

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        n=config.n_generate_sample,
        stop=config.stop,
        skip_special_tokens=False,
        truncate_prompt_tokens=config.max_model_len,
        seed=config.seed,
        logprobs=config.logprobs
    )
    if config.api:
        llm = MultiAPIInference(config)
    else:
        llm = MultiGPUInference(config, sampling_params)
    dataset = create_prompt(config)
    prompts = [i["_prompt"] for i in dataset]
    outputs = llm.generate(prompts)
    if config.logprobs is not None:
        logprobs = [[[[(i.logprob, i.decoded_token) for i in sorted(logprob.values(), key=lambda x: x.rank)] \
            for logprob in out.logprobs] \
            for out in output.outputs] \
            for output in outputs]
    if not config.api:
        outputs = [[i.text for i in output.outputs] for output in outputs]
    for cnt, it in enumerate(outputs):
        dataset[cnt]["_response"] = it
        if config.logprobs is not None:
            dataset[cnt]["_logprobs"] = logprobs[cnt]

    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists(f"output/{config.model_dir.strip('/').split('/')[-1]}"):
        os.makedirs(f"output/{config.model_dir.strip('/').split('/')[-1]}")
    with open(f"output/{config.model_dir.strip('/').split('/')[-1]}/{config.qaf.strip('/').split('/')[-1]}_{config.split}{config.save_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, sort_keys=True, indent=4)
    llm.exit()
