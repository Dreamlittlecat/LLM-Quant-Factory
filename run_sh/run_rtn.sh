# 配置参数
MODELS=(
     "opt-125m"
     "opt-350m"
    "opt-1.3b"
    "opt-2.7b"
    "opt-6.7b"
   "opt-13b"
)
MODEL_DIR="/data/xjh/model_weight/opt"
MODELS=(
    #  "llama3.2-1b"
    #  "llama3.2-3b"
    #  "llama3-8b"
     "llama-7b"
    # "llama2-7b"
    # "llama2-13b"

)
MODEL_DIR="/data/xjh/model_weight/llama"


MODEL_DIRS=("/data/xjh/model_weight/opt","/data/xjh/model_weight/llama")
BLOCKSIZES=(128)
DEVICE="cuda:1"
DATASET="c4"
LOG_DIR="./xjh_logs/rtn_${DATASET}"
N_BITS=2
# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

# 遍历配置
for BLOCKSIZE in "${BLOCKSIZES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        # 配置日志文件名
        LOG_FILE="$LOG_DIR/${MODEL}-bits${N_BITS}-rtn-quant"
        if [ "$BLOCKSIZE" -eq -1 ]; then
            LOG_FILE+="_noblock"
        else
            LOG_FILE+="_blocksize_${BLOCKSIZE}"
        fi

        # 启用 GPTQ
        echo "Running RTN for model $MODEL and blocksize $BLOCKSIZE N_bits${N_BITS}..."
        python3 run_rtn.py "$MODEL_DIR/$MODEL" "$DATASET" --blocksize "$BLOCKSIZE" --n_bits $N_BITS --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1

        # LOG_FILE="$LOG_DIR/${MODEL}-sdm-quant_hadamard-osr_${OSR}_first2"
        # echo "Running SDM with GPTQ for model $MODEL and blocksize $BLOCKSIZE OSR $OSR first2..."
        # python3 run_sdm.py "$MODEL_DIR/$MODEL" "$DATASET" --blocksize "$BLOCKSIZE" --osr $OSR $HADAMARD --first2 --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1

    done
done