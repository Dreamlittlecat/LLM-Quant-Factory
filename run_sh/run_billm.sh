# 配置参数
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
DEVICE="cuda:0"

BLOCKSIZES=(128 )


DATASET="c4"
LOG_DIR="./xjh_logs/billm_${DATASET}"

# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

# 遍历配置
for BLOCKSIZE in "${BLOCKSIZES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        # 配置日志文件名
        LOG_FILE="$LOG_DIR/$(basename "$MODEL")-billm-quant"
        if [ "$BLOCKSIZE" -eq -1 ]; then
            LOG_FILE+="_noblock"
        else
            LOG_FILE+="_blocksize_${BLOCKSIZE}"
        fi
        # 启用 GPTQ
        echo "Running BiLLM with GPTQ for model $MODEL and blocksize $BLOCKSIZE ..."
        python3 run_billm.py "$MODEL" "$DATASET" braq --blocksize "$BLOCKSIZE" --salient_metric hessian --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1

        # # 禁用 GPTQ
        # echo "Running SDM without GPTQ for model $MODEL and blocksize $BLOCKSIZE..."
        # python3 run_sdm.py "$MODEL_DIR/$MODEL" "$DATASET" --blocksize "$BLOCKSIZE" $HADAMARD --disable_gptq --device "$DEVICE" >> "${LOG_FILE}_disable_gptq.log" 2>&1
    done
done
