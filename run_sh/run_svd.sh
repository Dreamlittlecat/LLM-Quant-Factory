

# 配置参数
MODELS=(
    "opt-125m"
    "opt-350m"
    "opt-1.3b"
    "opt-2.7b"
    "opt-6.7b"
    "opt-13b"
)
BLOCKSIZES=(128 )
DEVICE="cuda:0"

MODEL_DIR="/data/xjh/model_weight/opt"
#DATASET="pileval"
DATASET="c4"
RESROUND=1
CLUSTER=-1
RANK=32
LOG_DIR="./xjh_logs/svd_${DATASET}"

# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

# 遍历配置
for BLOCKSIZE in "${BLOCKSIZES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        # 配置日志文件名
        LOG_FILE="$LOG_DIR/${MODEL}-svd-quant_resround_${RESROUND}_rank_${RANK}_cluster_${CLUSTER}"
        if [ "$BLOCKSIZE" -eq -1 ]; then
            LOG_FILE+="_noblock"
        else
            LOG_FILE+="_blocksize_${BLOCKSIZE}"
        fi

        # 启用 GPTQ
        echo "Running SVD quant for model $MODEL and blocksize $BLOCKSIZE..."
        python3 run_svd.py "$MODEL_DIR/$MODEL" "$DATASET" --blocksize "$BLOCKSIZE" --res_round ${RESROUND} --rank ${RANK} --cluster ${CLUSTER} --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1
    done
done