

# 配置参数
# MODELS=(
#     "opt-125m"
#     "opt-350m"
#     "opt-1.3b"
#     "opt-2.7b"
#     "opt-6.7b"
#     "opt-13b"
# )
# MODEL_DIR="/data/xjh/model_weight/opt"

MODELS=(
    # "llama3.2-1b"
    # "llama3.2-3b"
    # "llama3-8b"
    "llama-7b"
    # "llama2-7b"
    # "llama2-13b"

)
MODEL_DIR="/data/xjh/model_weight/llama"
BLOCKSIZES=(128 )
DEVICE="cuda:1"


DATASET="c4"
LOG_DIR="./xjh_logs/pbllm_${DATASET}"
LOW_FRAC=0.90
# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

# 遍历配置
for BLOCKSIZE in "${BLOCKSIZES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        # 配置日志文件名
        LOG_FILE="$LOG_DIR/${MODEL}-pbllm-quant_low_frac_${LOW_FRAC}"
        if [ "$BLOCKSIZE" -eq -1 ]; then
            LOG_FILE+="_noblock"
        else
            LOG_FILE+="_blocksize_${BLOCKSIZE}"
        fi

        # 启用 GPTQ
        echo "Running PBLLM quant for model $MODEL and blocksize $BLOCKSIZE..."
        python3 run_pbllm.py "$MODEL_DIR/$MODEL" "$DATASET" xnor --low_frac "$LOW_FRAC" --high_bit 8 --salient_metric hessian --groupsize "$BLOCKSIZE" --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1
    done
done