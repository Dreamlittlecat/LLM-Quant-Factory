

MODELS=(

     
    "/data/xjh/model_weight/opt/opt-125m"
    "/data/xjh/model_weight/opt/opt-350m"
    "/data/xjh/model_weight/opt/opt-1.3b"
    "/data/xjh/model_weight/opt/opt-2.7b"
    "/data/xjh/model_weight/opt/opt-6.7b"
    "/data/xjh/model_weight/opt/opt-13b"

    "/data/xjh/model_weight/llama/llama3.2-1b"

    "/data/xjh/model_weight/llama/llama3.2-3b"
    "/data/xjh/model_weight/llama/llama3-8b"

    "/data/xjh/model_weight/llama/llama2-7b"

    "/data/xjh/model_weight/llama/llama2-13b"
    "/data/xjh/model_weight/llama/llama-7b"
     "/data/xjh/model_weight/llama/llama-13b"



    # "/data/xjh/model_weight/qwen/Qwen3-4B"
    # "/data/xjh/model_weight/qwen/Qwen3-8B"
    # "/data/xjh/model_weight/qwen/Qwen3-14B"
    # "/data/xjh/model_weight/qwen/Qwen3-0.6B-Base"
    # "/data/xjh/model_weight/qwen/Qwen3-1.7B-Base"
    # "/data/xjh/model_weight/qwen/Qwen3-4B-Base"
    # "/data/xjh/model_weight/qwen/Qwen3-8B-Base"
    # "/data/xjh/model_weight/qwen/Qwen3-14B-Base"



 )



DEVICE="cuda:1"

BLOCKSIZES=(128 )
DATASET="c4"
LOG_DIR="./xjh_logs/gptq_${DATASET}"
N_BITS=3
# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

# 遍历配置
for BLOCKSIZE in "${BLOCKSIZES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        # 配置日志文件名
        LOG_FILE="$LOG_DIR/$(basename "$MODEL")-bits${N_BITS}-gptq-quant"
        if [ "$BLOCKSIZE" -eq -1 ]; then
            LOG_FILE+="_noblock"
        else
            LOG_FILE+="_blocksize_${BLOCKSIZE}"
        fi

        # 启用 GPTQ
        echo "Running RTN for model $MODEL and blocksize $BLOCKSIZE N_bits${N_BITS}..."
        python3 run_gptq.py "$MODEL" "$DATASET" --blocksize "$BLOCKSIZE" --n_bits $N_BITS --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1

        # LOG_FILE="$LOG_DIR/${MODEL}-sdm-quant_hadamard-osr_${OSR}_first2"
        # echo "Running SDM with GPTQ for model $MODEL and blocksize $BLOCKSIZE OSR $OSR first2..."
        # python3 run_sdm.py "$MODEL_DIR/$MODEL" "$DATASET" --blocksize "$BLOCKSIZE" --osr $OSR $HADAMARD --first2 --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1

    done
done