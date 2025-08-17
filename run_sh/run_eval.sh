#!/bin/bash

# 创建日志目录（如果不存在）
LOG_DIR="xjh_logs/no_quant"
mkdir -p "$LOG_DIR"

# 定义模型数组
models=(
    # "/data/xjh/model_weight/llama/llama3.2-1b"
    # "/data/xjh/model_weight/llama/llama3.2-3b"
    # "/data/xjh/model_weight/llama/llama3-8b"
    # "/data/xjh/model_weight/llama/llama2-7b"
    # "/data/xjh/model_weight/llama/llama2-13b"
    # "/data/xjh/model_weight/llama/llama-7b"
    # "/data/xjh/model_weight/llama/llama-13b"
    "/data/xjh/model_weight/llama/vicuna-7b-1.1"
    # "/data/xjh/model_weight/opt/opt-125m"
    # "/data/xjh/model_weight/opt/opt-350m"
    # "/data/xjh/model_weight/opt/opt-1.3b"
    # "/data/xjh/model_weight/opt/opt-2.7b"
    # "/data/xjh/model_weight/opt/opt-6.7b"
    # "/data/xjh/model_weight/opt/opt-13b"
)

# 遍历模型并运行脚本
for model in "${models[@]}"; do
    model_name=$(basename "$model")
    echo "Running evaluation for model: $model_name"
    python eval_test.py --model "$model" >> "${LOG_DIR}/${model_name}.log" 2>&1
    echo "Completed evaluation for: $model_name"
done

echo "All evaluations completed!"