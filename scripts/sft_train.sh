export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29500}"

train_path=src/rea_rl/sft.py
model_name_or_path=$1
train_file=$2
output_dir=$3

# HOST_NUM will be 1
for i in {1..5}
do
    torchrun --nnodes 1 --nproc_per_node 2 \
        --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
        ${train_path} \
        --model_name_or_path ${model_name_or_path} \
        --deepspeed recipes/accelerate_configs/deepspeed_config_zero2.json \
        --train_file ${train_file} \
        --use_lora True \
        --lora_config recipes/accelerate_configs/lora_config.json \
        --preprocessing_num_workers 16 \
        --dataloader_num_workers 8 \
        --dataloader_pin_memory True \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 32 \
        --num_train_epochs 1 \
        --save_strategy "steps" \
        --save_steps 200 \
        --save_total_limit 5 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --block_size 8192 \
        --do_train \
        --evaluation_strategy "no" \
        --validation_split_percentage 0 \
        --bf16 \
        --bf16_full_eval \
        --ddp_timeout 3600 \
        --seed 1 \
        --gradient_checkpointing \
        --output_dir ${output_dir}
done
