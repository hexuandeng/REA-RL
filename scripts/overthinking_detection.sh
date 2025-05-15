datasets=("aime2024_test" "amc23_test" "gaokao2023en_test" "GSM8K_test" "math500_test" "olympiadbench_test")

for dataset in "${datasets[@]}"
do
    python -u src/generation/generate.py \
        --config recipes/generation/deepseek.yaml \
        --qaf dataset/DeepScaleR/deepscaler \
        --model_dir DeepSeek-R1-Distill-Qwen-7B \
        --max_model_len 16384 \
        --max_tokens 16384 \
        --split math_cot

    python -u src/generation/generate.py \
        --config recipes/generation/qwen.yaml \
        --qaf output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_7B_ \
        --model_dir Qwen2.5-32B-Instruct \
        --max_model_len 16384 \
        --max_tokens 8192 \
        --split labeling_distill_use_asw

    python -u src/generation/labeling_distill_utils.py \
        --file output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_use_asw.json \
        --model_dir Qwen2.5-32B-Instruct \
        --use_asw

    python -u src/generation/generate.py \
        --config recipes/generation/qwen.yaml \
        --qaf output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_use_asw_parse \
        --model_dir Qwen2.5-32B-Instruct \
        --temperature 0 \
        --max_tokens 2048 \
        --split json

    python -u src/generation/labeling_distill_utils.py \
        --file output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_use_asw_parse_json.json \
        --model_dir Qwen2.5-32B-Instruct \
        --clean

    python -u src/generation/generate.py \
        --config recipes/generation/qwen.yaml \
        --qaf output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_use_asw_parse_json_clean \
        --model_dir DeepSeek-R1-Distill-Qwen-7B \
        --max_tokens 1024 \
        --split math_cot

    python -u src/generation/math_verify_all.py \
        --input_file output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_7B__labeling_distill_use_asw_parse_json_clean_math_cot.json


    python -u src/generation/generate.py \
        --config recipes/generation/qwen.yaml \
        --qaf output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_7B_ \
        --model_dir Qwen2.5-32B-Instruct \
        --max_model_len 16384 \
        --max_tokens 8192 \
        --split labeling_distill

    python -u src/generation/labeling_distill_utils.py \
        --file output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill.json \
        --model_dir Qwen2.5-32B-Instruct

    python -u src/generation/generate.py \
        --config recipes/generation/qwen.yaml \
        --qaf output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_parse \
        --model_dir Qwen2.5-32B-Instruct \
        --temperature 0 \
        --max_tokens 2048 \
        --split json

    python -u src/generation/labeling_distill_utils.py \
        --file output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_parse_json.json \
        --model_dir Qwen2.5-32B-Instruct \
        --clean

    python -u src/generation/generate.py \
        --config recipes/generation/qwen.yaml \
        --qaf output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_parse_json_clean \
        --model_dir DeepSeek-R1-Distill-Qwen-7B \
        --max_tokens 1024 \
        --split math_cot

    python -u src/generation/math_verify_all.py \
        --input_file output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_7B__labeling_distill_parse_json_clean_math_cot.json

done

for dataset in "${datasets[@]}"
do
    python -u src/generation/generate.py \
        --config recipes/generation/qwen.yaml \
        --qaf output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_7B_ \
        --model_dir Qwen2.5-32B-Instruct \
        --max_model_len 8192 \
        --max_tokens 8192 \
        --split labeling_distill_use_asw \
        --save_suffix _8k

    python -u src/generation/labeling_distill_utils.py \
        --file output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_use_asw_8k.json \
        --model_dir Qwen2.5-32B-Instruct \
        --use_asw

    python -u src/generation/generate.py \
        --config recipes/generation/qwen.yaml \
        --qaf output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_use_asw_8k_parse \
        --model_dir Qwen2.5-32B-Instruct \
        --temperature 0 \
        --max_tokens 2048 \
        --split json

    python -u src/generation/labeling_distill_utils.py \
        --file output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_use_asw_8k_parse_json.json \
        --model_dir Qwen2.5-32B-Instruct \
        --clean

    python -u src/generation/generate.py \
        --config recipes/generation/qwen.yaml \
        --qaf output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_use_asw_8k_parse_json_clean \
        --model_dir DeepSeek-R1-Distill-Qwen-7B \
        --max_tokens 1024 \
        --split math_cot

    python -u src/generation/math_verify_all.py \
        --input_file output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_7B__labeling_distill_use_asw_8k_parse_json_clean_math_cot.json &


    python -u src/generation/generate.py \
        --config recipes/generation/qwen.yaml \
        --qaf output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_7B_ \
        --model_dir Qwen2.5-32B-Instruct \
        --max_model_len 8192 \
        --max_tokens 8192 \
        --split labeling_distill \
        --save_suffix _8k

    python -u src/generation/labeling_distill_utils.py \
        --file output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_8k.json \
        --model_dir Qwen2.5-32B-Instruct

    python -u src/generation/generate.py \
        --config recipes/generation/qwen.yaml \
        --qaf output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_8k_parse \
        --model_dir Qwen2.5-32B-Instruct \
        --temperature 0 \
        --max_tokens 2048 \
        --split json

    python -u src/generation/labeling_distill_utils.py \
        --file output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_8k_parse_json.json \
        --model_dir Qwen2.5-32B-Instruct \
        --clean

    python -u src/generation/generate.py \
        --config recipes/generation/qwen.yaml \
        --qaf output/Qwen2.5-32B-Instruct/${dataset}_math_cot_7B__labeling_distill_8k_parse_json_clean \
        --model_dir DeepSeek-R1-Distill-Qwen-7B \
        --max_tokens 1024 \
        --split math_cot

    python -u src/generation/math_verify_all.py \
        --input_file output/DeepSeek-R1-Distill-Qwen-7B/${dataset}_math_cot_7B__labeling_distill_8k_parse_json_clean_math_cot.json

done
