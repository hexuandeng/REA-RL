export NCCL_P2P_DISABLE=1

NAME=Gen4-Base
ROOT=model/DeepSeek-R1-Distill-Qwen-7B-${NAME}
VLLM_PORT=12342
Group_PORT=51212

for i in {1..5}
do
    kill -9 $(ps aux | grep 'vllm_server' | grep ${VLLM_PORT} | awk '{print $2}')

    bash scripts/vllm.sh ${VLLM_PORT} 2

    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
        --gpu_ids 0,1 --num_processes=2 --main_process_port 29050 src/rea_rl/grpo.py \
        --config recipes/DeepSeek-R1-Distill-Qwen-7B/config_GRPO.yaml \
        --num_generations 4 \
        --output_dir ${ROOT} \
        --vllm_server_port ${VLLM_PORT} \
        --group_port ${Group_PORT}
done
