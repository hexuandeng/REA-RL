# Downlrad dataset from https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset

python -u src/generation/generate.py \
    --config recipes/generation/deepseek.yaml \
    --qaf dataset/DeepScaleR/deepscaler \
    --model_dir DeepSeek-R1-Distill-Qwen-7B \
    --max_model_len 8192 \
    --max_tokens 8192 \
    --split math_cot

python -u src/generation/generate.py \
    --config recipes/generation/qwen.yaml \
    --qaf output/DeepSeek-R1-Distill-Qwen-7B/deepscaler \
    --model_dir Qwen2.5-32B-Instruct \
    --max_model_len 16384 \
    --max_tokens 8192 \
    --split labeling_distill_use_asw

python -u src/generation/labeling_distill_utils.py \
    --file output/Qwen2.5-32B-Instruct/deepscaler_labeling_distill_use_asw.json \
    --model_dir Qwen2.5-32B-Instruct \
    --use_asw

python -u src/generation/generate.py \
    --config recipes/generation/qwen.yaml \
    --qaf output/Qwen2.5-32B-Instruct/deepscaler_labeling_distill_use_asw_parse \
    --model_dir Qwen2.5-32B-Instruct \
    --temperature 0 \
    --max_tokens 2048 \
    --split json

python -u src/generation/labeling_distill_utils.py \
    --file output/Qwen2.5-32B-Instruct/deepscaler_labeling_distill_use_asw_parse_json.json \
    --model_dir Qwen2.5-32B-Instruct \
    --train_file dataset/deepscaler_reflection_training.json \
    --clean

# Baseline Training Data Generation

python -u src/generation/generate.py \
    --config recipes/generation/qwen.yaml \
    --qaf output/Qwen2.5-32B-Instruct/deepscaler_labeling_distill_use_asw_parse_json_clean \
    --model_dir DeepSeek-R1-Distill-Qwen-7B \
    --max_tokens 1024 \
    --split math_cot

python -u src/generation/math_verify_all.py \
    --input_file output/DeepSeek-R1-Distill-Qwen-7B/deepscaler_labeling_distill_use_asw_parse_json_clean_math_cot.json \
    --judge_output_file output/DeepSeek-R1-Distill-Qwen-7B/deepscaler_labeling_distill_judge.json

python -u src/generation/sft_data.py
