export CUDA_VISIBLE_DEVICES=$1

datasets=("aime2024_test" "amc23_test" "gaokao2023en_test" "GSM8K_test" "math500_test" "olympiadbench_test")
NAME=$2
CKPT=$3

for dataset in "${datasets[@]}"
do
    if [[ "$dataset" == "aime2024_test" ]] || [[ "$dataset" == "amc23_test" ]]; then
        num_samples=8
    else
        num_samples=1
    fi

    if [[ "$NAME" == "7B" ]]; then
        python -u src/generation/generate.py \
            --config recipes/generation/deepseek.yaml \
            --qaf dataset/eval_data/${dataset} \
            --model_dir DeepSeek-R1-Distill-Qwen-7B \
            --split math_cot \
            --max_tokens 16384 \
            --max_model_len 16384 \
            --n_generate_sample ${num_samples} \
            --save_suffix _${NAME}_${CKPT}
    else
        python -u src/generation/generate.py \
            --config recipes/generation/deepseek.yaml \
            --qaf dataset/eval_data/${dataset} \
            --model_dir DeepSeek-R1-Distill-Qwen-7B \
            --lora model/DeepSeek-R1-Distill-Qwen-7B-${NAME}/checkpoint-${CKPT} \
            --split math_cot \
            --max_tokens 16384 \
            --max_model_len 16384 \
            --n_generate_sample ${num_samples} \
            --save_suffix _${NAME}_${CKPT}
    fi

    echo "{$dataset}_${NAME}_${CKPT}"
    python -u src/generation/math_verify_all.py \
        --input_file output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_${NAME}_${CKPT}.json \
        --max_length 8192 &

    python -u src/generation/math_verify_all.py \
        --input_file output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_${NAME}_${CKPT}.json \
        --max_length 16384 &

done

wait
