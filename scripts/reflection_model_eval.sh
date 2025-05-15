datasets=("aime2024_test" "amc23_test" "gaokao2023en_test" "GSM8K_test" "math500_test" "olympiadbench_test")

for dataset in "${datasets[@]}"
do
    python -u src/generation/generate.py \
        --config recipes/generation/deepseek.yaml \
        --qaf dataset/eval_data/${dataset} \
        --model_dir DeepSeek-R1-Distill-Qwen-7B \
        --split math_cot

    python -u src/generation/generate.py \
        --config recipes/generation/qwen.yaml \
        --qaf output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_7B_ \
        --model_dir Qwen2.5-7B-Instruct \
        --split labeling \
        --max_model_len 16384 \
        --lora model/Qwen2.5-7B-Labeling/checkpoint-600/adapter_model \
        --logprobs 5 \
        --save_suffix _16k

    python -u src/generation/labeling_to_prefix.py \
        --qaf output/Qwen2.5-7B-Instruct/${dataset}_math_cot_7B__labeling_16k \
        --model_dir DeepSeek-R1-Distill-Qwen-7B

    python -u src/generation/generate.py \
        --config recipes/generation/qwen.yaml \
        --qaf output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_7B_ \
        --model_dir Qwen2.5-7B-Instruct \
        --split labeling \
        --max_model_len 8192 \
        --lora model/Qwen2.5-7B-Labeling/checkpoint-600/adapter_model \
        --logprobs 5 \
        --save_suffix _8k

    python -u src/generation/labeling_to_prefix.py \
        --qaf output/Qwen2.5-7B-Instruct/${dataset}_math_cot_7B__labeling_8k \
        --model_dir DeepSeek-R1-Distill-Qwen-7B
done

suffix=("Weak" "Normal" "Strong")
type=("" "_Rand")

for dataset in "${datasets[@]}"
do
    for s in "${suffix[@]}"
    do
        for t in "${type[@]}"
        do
            python -u src/generation/generate.py \
                --config recipes/generation/deepseek.yaml \
                --qaf output/Qwen2.5-7B-Instruct/${dataset}_math_cot_7B__labeling_16k_${s}${t} \
                --model_dir DeepSeek-R1-Distill-Qwen-7B \
                --max_tokens 1024 \
                --split math_cot
            
            python -u src/generation/math_verify_all.py \
                --input_file output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_7B__labeling_16k_${s}${t}_math_cot.json &

            python -u src/generation/generate.py \
                --config recipes/generation/deepseek.yaml \
                --qaf output/Qwen2.5-7B-Instruct/${dataset}_math_cot_7B__labeling_8k_${s}${t} \
                --model_dir DeepSeek-R1-Distill-Qwen-7B \
                --max_tokens 1024 \
                --split math_cot
            
            python -u src/generation/math_verify_all.py \
                --input_file output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_7B__labeling_8k_${s}${t}_math_cot.json &
        done
    done
done

suffix=("0.7" "0.8" "0.9")
type=("_Rand")

for dataset in "${datasets[@]}"
do
    for s in "${suffix[@]}"
    do
        for t in "${type[@]}"
        do
            python -u src/generation/generate.py \
                --config recipes/generation/deepseek.yaml \
                --qaf output/Qwen2.5-7B-Instruct/${dataset}_math_cot_7B__labeling_16k_${s}${t} \
                --model_dir DeepSeek-R1-Distill-Qwen-7B \
                --max_tokens 1024 \
                --split math_cot
            
            python -u src/generation/math_verify_all.py \
                --input_file output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_7B__labeling_16k_${s}${t}_math_cot.json &

            python -u src/generation/generate.py \
                --config recipes/generation/deepseek.yaml \
                --qaf output/Qwen2.5-7B-Instruct/${dataset}_math_cot_7B__labeling_8k_${s}${t} \
                --model_dir DeepSeek-R1-Distill-Qwen-7B \
                --max_tokens 1024 \
                --split math_cot
            
            python -u src/generation/math_verify_all.py \
                --input_file output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_7B__labeling_8k_${s}${t}_math_cot.json &
        done
    done
done

wait
