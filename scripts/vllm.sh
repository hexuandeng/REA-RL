#!/usr/bin/env bash

PORT=${1}

echo $(ps aux | grep 'vllm_server' | grep ${PORT} | awk '{print $2}')
kill -9 $(ps aux | grep 'vllm_server' | grep ${PORT} | awk '{print $2}')

sleep 5

CUDA_VISIBLE_DEVICES=${2} nohup python -u src/rea_rl/vllm_server.py \
    --model DeepSeek-R1-Distill-Qwen-7B \
    --tensor_parallel_size 1 \
    --host 0.0.0.0 \
    --port ${PORT} \
    --gpu_memory_utilization 0.5 \
    --dtype bfloat16 \
    --max_model_len 10240 \
    --enable_prefix_caching True > logs/trl_${PORT}.log 2>&1 &
